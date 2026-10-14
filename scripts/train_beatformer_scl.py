"""Self-Supervised Contrastive Learning (SCL) Training Script for BeatFormer.

Synthesized from BeatFormer (Comas & Sukno, ICCVW 2025) and RePSS (Zhang et al., 2024).
Trains BeatFormer on contrastive triplets (anchor, augmented positive, resampled negative)
using Earth Mover's Distance (EMD), Orthonormal regularizer, and Spectral Sparsity.
Saves checkpoint to artifacts/beatformer_scl.pt.
"""

import os
import glob
import time
import cv2
import numpy as np
import torch
import torch.optim as optim

from models.hybrid.beatformer import BeatFormer
from models.hybrid.scl_pretrain import (
    SCLAugmentations,
    temporal_resample,
    emd_loss,
    scl_triplet_loss,
    spectral_sparsity_loss,
)
from preprocessing.roi_extractor import ROIExtractor


def extract_or_generate_traces(num_windows: int = 120, win_len: int = 150) -> np.ndarray:
    """Collect real facial RGB traces from UBFC-rPPG dataset or synthesize realistic traces.

    Returns:
        traces: (N, win_len, 3) float32 numpy array.
    """
    collected = []
    ubfc_dirs = sorted(glob.glob("datasets/ubfc/subject*"))
    extractor = ROIExtractor()

    print(f"Discovered {len(ubfc_dirs)} UBFC subjects for SCL pretraining.")
    # Extract from first 6 subjects to keep data collection quick
    for sdir in ubfc_dirs[:6]:
        vpath = os.path.join(sdir, "vid.mp4")
        if not os.path.exists(vpath):
            continue
        cap = cv2.VideoCapture(vpath)
        frames = []
        for _ in range(300):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) >= win_len:
            out = extractor.process_video(np.array(frames))
            rgb = out["rgb_signals"]  # (T, 3)
            for start in range(0, len(rgb) - win_len + 1, 30):
                w = rgb[start : start + win_len]
                # Normalize by temporal mean
                w_norm = (w / (np.mean(w, axis=0, keepdims=True) + 1e-8)) - 1.0
                collected.append(w_norm)
                if len(collected) >= num_windows:
                    break
        if len(collected) >= num_windows:
            break

    extractor.close()

    # If collected less than requested, synthesize physiological traces
    remaining = num_windows - len(collected)
    if remaining > 0:
        fs = 30.0
        t = np.arange(win_len) / fs
        for _ in range(remaining):
            bpm = np.random.uniform(50.0, 140.0)
            f0 = bpm / 60.0
            pulse = np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
            # Hemoglobin absorption in Green channel
            r = 0.005 * pulse + np.random.normal(0, 0.002, win_len)
            g = -0.020 * pulse + np.random.normal(0, 0.002, win_len)
            b = 0.003 * pulse + np.random.normal(0, 0.002, win_len)
            collected.append(np.column_stack([r, g, b]))

    return np.array(collected, dtype=np.float32)


def train_beatformer(
    output_path: str = "artifacts/beatformer_scl.pt",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
):
    """Train BeatFormer with SCL triplet and orthonormal loss."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Handle sm_120 or incompatible CUDA architectures gracefully by falling back to CPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            # Test small kernel
            _ = torch.zeros(1).cuda() + 1
            device = torch.device("cuda")
        except Exception:
            device = torch.device("cpu")
    print(f"Training BeatFormer on device: {device}")

    traces = extract_or_generate_traces(num_windows=160, win_len=150)
    print(f"Dataset prepared: {traces.shape[0]} windows of length {traces.shape[1]}")

    model = BeatFormer(window_len=150, num_bins=128, fs=30.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_samples = len(traces)
    indices = np.arange(n_samples)

    start_time = time.time()
    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for b_start in range(0, n_samples - batch_size + 1, batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            batch_anchor = traces[b_idx]  # (B, L, 3)

            # Generate positive view: photometric color noise & slight jitter
            batch_pos = np.copy(batch_anchor)
            jitter_scale = np.random.uniform(0.85, 1.15, (batch_size, 1, 3))
            batch_pos = batch_pos * jitter_scale + np.random.normal(0, 0.001, batch_pos.shape)

            # Generate negative view: temporal resampling factor (altering heart rate)
            batch_neg = np.zeros_like(batch_anchor)
            for i in range(batch_size):
                factor = np.random.choice([0.75, 1.35])
                batch_neg[i] = temporal_resample(batch_anchor[i], factor)

            anc_tensor = torch.from_numpy(batch_anchor).to(device)
            pos_tensor = torch.from_numpy(batch_pos.astype(np.float32)).to(device)
            neg_tensor = torch.from_numpy(batch_neg.astype(np.float32)).to(device)

            optimizer.zero_grad()
            out_anc = model(anc_tensor)
            out_pos = model(pos_tensor)
            out_neg = model(neg_tensor)

            # SCL Triplet Loss with Earth Mover's Distance
            loss_scl = scl_triplet_loss(out_anc["psd"], out_pos["psd"], out_neg["psd"], gamma=0.5)

            # Orthonormal projection loss (BeatFormer Eq. 10)
            loss_ortho = out_anc["ortho_loss"] + out_pos["ortho_loss"]

            # Spectral sparsity regularizer (Zhang et al. 2024)
            loss_sparse = spectral_sparsity_loss(out_anc["psd"])

            loss_total = loss_scl + 0.1 * loss_ortho + 0.05 * loss_sparse
            loss_total.backward()
            optimizer.step()

            epoch_loss += loss_total.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} (SCL: {loss_scl.item():.4f}, Ortho: {loss_ortho.item():.4f})")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f}s. Saving weights to {output_path}...")
    torch.save(model.state_dict(), output_path)
    print(f"Model successfully saved. Size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    train_beatformer()
