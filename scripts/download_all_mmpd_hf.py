#!/usr/bin/env python3
"""Automated Sequential Downloader & Transcoder for MMPD from Hugging Face.

Downloads selected subjects from WeiQian98/MMPD, transcodes the 4D raw RGB video
buffer (~430-600 MB) to lightweight H.264 MP4 (~2-4 MB, 99.4% size reduction),
extracts and saves ground-truth PPG and metadata in a lightweight .mat file (~15 KB),
and immediately deletes the massive raw .mat to keep peak disk usage under 600 MB.

Default target cohort: 4 subjects stratified across all 4 Fitzpatrick skin types in MMPD:
- p3: Fitzpatrick Type III (Fair/Medium)
- p1: Fitzpatrick Type IV  (Olive/Moderate Brown)
- p7: Fitzpatrick Type V   (Brown/Dark)
- p8: Fitzpatrick Type VI  (Deeply Pigmented Dark)
Across all 20 scenarios per subject = 80 total trials (~220-300 MB total final disk size).
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
import urllib.request
import scipy.io as sio
import numpy as np
import cv2

from estimation.czt_estimator import CZTHRTransformer


DEFAULT_SUBJECTS = ["p3", "p1", "p7", "p8"]
HF_REPO_URL = "https://huggingface.co/datasets/WeiQian98/MMPD/resolve/main"


def get_hf_token() -> str:
    """Retrieve Hugging Face token from environment or cache."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    cache_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return f.read().strip()
    raise RuntimeError("No Hugging Face token found in environment or ~/.cache/huggingface/token")


def transcode_video_array(
    video_array: np.ndarray,
    out_mp4_path: Path,
    fps: float = 30.0,
    crf: int = 18
) -> bool:
    """Transcode a 4D numpy array (T, dim1, dim2, 3) to H.264 MP4 via ffmpeg."""
    if video_array.dtype != np.uint8:
        if video_array.max() <= 1.0:
            video_array = (video_array * 255.0).clip(0, 255).astype(np.uint8)
        else:
            video_array = video_array.clip(0, 255).astype(np.uint8)

    T, dim1, dim2, C = video_array.shape

    # Pipe raw video frames directly into ffmpeg stdin
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{dim2}x{dim1}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(out_mp4_path)
    ]

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.communicate(input=video_array.tobytes())
        return proc.returncode == 0 and out_mp4_path.exists() and out_mp4_path.stat().st_size > 10_000
    except Exception as e:
        print(f"      [!] FFmpeg transcoding error: {e}")
        return False


def process_single_trial(
    subj: str,
    scenario_idx: int,
    dest_dir: Path,
    hf_token: str,
    crf: int = 18
) -> bool:
    """Download, transcode, strip raw video, and clean up a single MMPD trial."""
    subj_dir = dest_dir / subj
    subj_dir.mkdir(parents=True, exist_ok=True)

    file_stem = f"{subj}_{scenario_idx}"
    target_mp4 = subj_dir / f"{file_stem}.mp4"
    target_mat = subj_dir / f"{file_stem}.mat"
    temp_raw_mat = subj_dir / f"{file_stem}_raw.mat"

    # Skip if already processed and valid
    if target_mp4.exists() and target_mat.exists() and target_mp4.stat().st_size > 100_000:
        print(f"  [-] {file_stem} already completed ({target_mp4.stat().st_size / 1e6:.2f} MB). Skipping.")
        return True

    file_name = f"{file_stem}.mat"
    url = f"{HF_REPO_URL}/{file_name}"
    print(f"  [*] Downloading {file_name} from Hugging Face...")
    t0 = time.time()

    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {hf_token}", "User-Agent": "Mozilla/5.0"}
    )

    downloaded = False
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(temp_raw_mat, "wb") as out_f:
                while chunk := resp.read(16 * 1024 * 1024):
                    out_f.write(chunk)
            downloaded = True
            break
        except Exception as e:
            print(f"      [!] Attempt {attempt}/3 failed for {file_name}: {e}")
            if temp_raw_mat.exists():
                temp_raw_mat.unlink()
            time.sleep(3)

    if not downloaded:
        print(f"      [!] Failed to download {file_name} after 3 attempts.")
        return False

    dl_time = time.time() - t0
    raw_size_mb = temp_raw_mat.stat().st_size / 1e6
    print(f"      Downloaded {raw_size_mb:.1f} MB in {dl_time:.1f}s ({raw_size_mb/max(dl_time, 0.1):.1f} MB/s).")

    # Load MAT and extract video + metadata
    try:
        mat_data = sio.loadmat(str(temp_raw_mat))
    except Exception as e:
        print(f"      [!] Failed to parse MAT file {file_name}: {e}")
        temp_raw_mat.unlink()
        return False

    if "video" not in mat_data:
        print(f"      [!] No 'video' key in {file_name}.")
        temp_raw_mat.unlink()
        return False

    raw_video = mat_data["video"]
    t1 = time.time()

    # Transcode video array to H.264 MP4
    ok = transcode_video_array(raw_video, target_mp4, fps=30.0, crf=crf)
    if not ok:
        print(f"      [!] Transcoding failed for {file_name}.")
        if temp_raw_mat.exists():
            temp_raw_mat.unlink()
        if target_mp4.exists():
            target_mp4.unlink()
        return False

    tc_time = time.time() - t1
    mp4_size_mb = target_mp4.stat().st_size / 1e6
    compression_ratio = 100.0 * (1.0 - mp4_size_mb / raw_size_mb)
    print(f"      Transcoded to {mp4_size_mb:.2f} MB in {tc_time:.1f}s ({compression_ratio:.1f}% space saved).")

    # Build clean metadata dictionary
    meta = {k: mat_data[k] for k in mat_data if k != "video" and not k.startswith("__")}
    meta["fs"] = 30.0

    # Ensure ground truth heart rate is computed via 512-bin CZT if not directly present
    ppg = None
    for k in ["GT_ppg", "d_ppg", "ppg", "PPG", "bvp"]:
        if k in mat_data:
            ppg = np.squeeze(mat_data[k]).astype(np.float64)
            break

    if ppg is not None and len(ppg) >= 128:
        czt = CZTHRTransformer(fs=30.0, low_hz=0.7, high_hz=3.0, num_bins=512)
        meta["gt_HR"] = float(czt.estimate_hr(ppg))
        meta["d_ppg"] = ppg

    # Save lightweight metadata MAT file
    sio.savemat(str(target_mat), meta)

    # Immediately delete raw 400+ MB MAT file
    temp_raw_mat.unlink()
    print(f"      [+] Cleaned raw .mat. Saved {target_mp4.name} & {target_mat.name}.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sequentially download and transcode MMPD from HuggingFace")
    parser.add_argument("--dest_dir", type=str, default="data/MMPD", help="Target MMPD directory")
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS, help="List of subjects to process")
    parser.add_argument("--num_scenarios", type=int, default=20, help="Number of scenarios per subject (max 20)")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF quality level (default: 18)")
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    hf_token = get_hf_token()

    print("=" * 80)
    print("MMPD HUGGING FACE SEQUENTIAL DOWNLOADER & H.264 TRANSCODER")
    print(f"Target Cohort: {args.subjects} ({len(args.subjects)} subjects x {args.num_scenarios} scenarios)")
    print(f"Destination: {dest_dir.resolve()}")
    print("=" * 80)

    total_tasks = len(args.subjects) * args.num_scenarios
    completed = 0
    t_start = time.time()

    for s_idx, subj in enumerate(args.subjects, 1):
        print(f"\n==================== [{s_idx}/{len(args.subjects)}] Processing Subject: {subj} ====================")
        for sc_idx in range(args.num_scenarios):
            completed += 1
            print(f"\n({completed}/{total_tasks}) [{subj} - Scenario {sc_idx}/19]:")
            process_single_trial(
                subj=subj,
                scenario_idx=sc_idx,
                dest_dir=dest_dir,
                hf_token=hf_token,
                crf=args.crf
            )

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"[+] All {completed} trials processed successfully in {elapsed/60:.1f} minutes!")
    print(f"[+] Total disk usage in {dest_dir}:")
    subprocess.run(["du", "-sh", str(dest_dir)])
    print("=" * 80)


if __name__ == "__main__":
    main()
