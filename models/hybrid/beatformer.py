"""BeatFormer: Physiologically-Constrained Spectral Transformer.

Synthesized from BeatFormer (Martinez et al., ICCVW 2025; Comas & Sukno, 2025)
and RePSS (Zhang et al., 2024).

Architecture:
1. CZTLayer: Differentiable Chirp Z-Transform mapping RGB traces (L=150) to complex cardiac spectrum (M=128).
2. ZOCA: Zoomed Orthonormal Complex Attention with fixed illumination row, learned chrominance projections,
   Min-Max-Norm scaling (Eq. 7), and 8-term complex attention expansion (Eq. 9).
3. ESD: Energy Spectral Density feed-forward with true zero-anchored spectral gating (Eqs. 11-12).
4. ICZTLayer: Regularized pseudoinverse time-domain BVP waveform reconstructor.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F


class CZTLayer(nn.Module):
    """Differentiable Chirp Z-Transform layer for cardiac frequency zooming [0.7, 3.0] Hz."""

    def __init__(
        self,
        length: int = 150,
        num_bins: int = 128,
        fs: float = 30.0,
        low_hz: float = 0.7,
        high_hz: float = 3.0,
    ):
        super().__init__()
        self.length = length
        self.num_bins = num_bins
        self.fs = fs
        self.low_hz = low_hz
        self.high_hz = high_hz

        # Precompute CZT Vandermonde matrix: V[k, n] = A^(-n) * W^(k*n)
        n = np.arange(length, dtype=np.float64)
        k = np.arange(num_bins, dtype=np.float64)
        A = np.exp(1j * 2.0 * np.pi * low_hz / fs)
        W = np.exp(-1j * 2.0 * np.pi * (high_hz - low_hz) / ((num_bins - 1) * fs))

        An = A ** (-n)
        Wkn = W ** (np.outer(k, n))
        V_czt = Wkn * An[None, :]  # shape (M, L)

        self.register_buffer("v_re", torch.from_numpy(np.real(V_czt).astype(np.float32)))
        self.register_buffer("v_im", torch.from_numpy(np.imag(V_czt).astype(np.float32)))

        freq_axis = low_hz + k * ((high_hz - low_hz) / (num_bins - 1))
        self.register_buffer("freq_axis", torch.from_numpy(freq_axis.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute complex CZT spectrum."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        f_re = torch.einsum("blc,ml->bmc", x, self.v_re)
        f_im = torch.einsum("blc,ml->bmc", x, self.v_im)
        return f_re, f_im


class ICZTLayer(nn.Module):
    """Differentiable Inverse Chirp Z-Transform reconstructor using regularized pseudo-inverse."""

    def __init__(
        self,
        length: int = 150,
        num_bins: int = 128,
        fs: float = 30.0,
        low_hz: float = 0.7,
        high_hz: float = 3.0,
        reg_lambda: float = 1e-4,
    ):
        super().__init__()
        self.length = length
        self.num_bins = num_bins

        n = np.arange(length, dtype=np.float64)
        k = np.arange(num_bins, dtype=np.float64)
        A = np.exp(1j * 2.0 * np.pi * low_hz / fs)
        W = np.exp(-1j * 2.0 * np.pi * (high_hz - low_hz) / ((num_bins - 1) * fs))

        An = A ** (-n)
        Wkn = W ** (np.outer(k, n))
        V_czt = Wkn * An[None, :]  # shape (M, L)

        VH = V_czt.conj().T
        inv_mat = scipy.linalg.inv(VH @ V_czt + reg_lambda * np.eye(length))
        V_iczt = inv_mat @ VH  # shape (L, M)

        self.register_buffer("v_iczt_re", torch.from_numpy(np.real(V_iczt).astype(np.float32)))
        self.register_buffer("v_iczt_im", torch.from_numpy(np.imag(V_iczt).astype(np.float32)))

    def forward(self, f_re: torch.Tensor, f_im: torch.Tensor) -> torch.Tensor:
        """Reconstruct real time-domain signal."""
        if f_re.dim() == 2:
            f_re = f_re.unsqueeze(-1)
            f_im = f_im.unsqueeze(-1)
        recon_re = torch.einsum("bmc,lm->blc", f_re, self.v_iczt_re)
        recon_im = torch.einsum("bmc,lm->blc", f_im, self.v_iczt_im)
        return recon_re - recon_im


class OrthonormalProjection(nn.Module):
    """3x3 projection with fixed illumination row 0 and 2 learned orthogonal chrominance rows."""

    def __init__(self):
        super().__init__()
        # Fixed illumination row: [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
        self.register_buffer("row0", torch.full((1, 3), 1.0 / np.sqrt(3.0), dtype=torch.float32))
        # Initialized orthogonal chrominance vectors (POS / CHROM informed)
        init_r1 = torch.tensor([[0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]], dtype=torch.float32)
        init_r2 = torch.tensor([[-2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)]], dtype=torch.float32)
        self.rows12 = nn.Parameter(torch.cat([init_r1, init_r2], dim=0))

    def get_matrix(self) -> torch.Tensor:
        """Return full 3x3 orthonormal projection matrix."""
        return torch.cat([self.row0, self.rows12], dim=0)

    def orthonormal_loss(self) -> torch.Tensor:
        """Loss penalizing deviation from orthonormality (BeatFormer Eq. 10)."""
        W = self.get_matrix()
        a0, a1, a2 = W[0], W[1], W[2]
        dot01 = torch.sum(a0 * a1)
        dot02 = torch.sum(a0 * a2)
        dot12 = torch.sum(a1 * a2)
        norm1 = torch.norm(a1) - 1.0
        norm2 = torch.norm(a2) - 1.0
        return dot01 ** 2 + dot02 ** 2 + dot12 ** 2 + norm1 ** 2 + norm2 ** 2


class ZOCABlock(nn.Module):
    """Zoomed Orthonormal Complex Attention Block (BeatFormer Eqs. 6-10)."""

    def __init__(self, num_bins: int = 128, channels: int = 3):
        super().__init__()
        self.num_bins = num_bins
        self.channels = channels
        self.pos_emb = nn.Parameter(torch.randn(1, num_bins, channels) * 0.02)
        self.proj_q = OrthonormalProjection()
        self.proj_k = OrthonormalProjection()
        self.proj_v = OrthonormalProjection()

        self.scale = 1.0 / np.sqrt(channels)

    @staticmethod
    def _min_max_norm(mat: torch.Tensor) -> torch.Tensor:
        """Min-Max-Norm attention scaling (BeatFormer Eq. 7)."""
        min_v = mat.amin(dim=-1, keepdim=True)
        max_v = mat.amax(dim=-1, keepdim=True)
        normed = (mat - min_v) / (max_v - min_v + 1e-8)
        return normed / (normed.sum(dim=-1, keepdim=True) + 1e-8)

    def forward(self, f_re: torch.Tensor, f_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 8-term complex self-attention over spectral representations (BeatFormer Eq. 9)."""
        x_re = f_re + self.pos_emb
        x_im = f_im

        W_q = self.proj_q.get_matrix()
        W_k = self.proj_k.get_matrix()
        W_v = self.proj_v.get_matrix()

        Q_re = torch.matmul(x_re, W_q)
        Q_im = torch.matmul(x_im, W_q)
        K_re = torch.matmul(x_re, W_k)
        K_im = torch.matmul(x_im, W_k)
        V_re = torch.matmul(x_re, W_v)
        V_im = torch.matmul(x_im, W_v)

        # 8-term complex attention expansion (BeatFormer Eq. 9)
        A_rr = self._min_max_norm(torch.bmm(Q_re, K_re.transpose(1, 2)) * self.scale)
        A_ii = self._min_max_norm(torch.bmm(Q_im, K_im.transpose(1, 2)) * self.scale)
        A_ri = self._min_max_norm(torch.bmm(Q_re, K_im.transpose(1, 2)) * self.scale)
        A_ir = self._min_max_norm(torch.bmm(Q_im, K_re.transpose(1, 2)) * self.scale)

        real_part = (
            torch.bmm(A_rr, V_re)
            - torch.bmm(A_ii, V_re)
            - torch.bmm(A_ri, V_im)
            - torch.bmm(A_ir, V_im)
        ) * 0.5
        imag_part = (
            torch.bmm(A_rr, V_im)
            - torch.bmm(A_ii, V_im)
            + torch.bmm(A_ri, V_re)
            + torch.bmm(A_ir, V_re)
        ) * 0.5

        out_re = real_part + f_re
        out_im = imag_part + f_im
        return out_re, out_im


class ESDBlock(nn.Module):
    """Energy Spectral Density Feed-Forward Block (BeatFormer Eqs. 11-12)."""

    def __init__(self, num_bins: int = 128, channels: int = 3, hidden_dim: int = 48):
        super().__init__()
        in_dim = channels * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.channel_fusion = nn.Linear(channels, 1, bias=False)
        with torch.no_grad():
            self.channel_fusion.weight.copy_(torch.tensor([[-0.2, 1.0, -0.2]]))

    def forward(
        self,
        z_re: torch.Tensor,
        z_im: torch.Tensor,
        f_re: torch.Tensor,
        f_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ESD gated spectrum and fuse channels (Eqs. 11-12)."""
        feat = torch.cat([z_re, z_im], dim=-1)  # (B, M, 6)
        z_prime = self.mlp(feat)  # (B, M, 2)
        zp_re = z_prime[..., 0:1]
        zp_im = z_prime[..., 1:2]

        # FIXED: Zero-anchored energy contribution weight S(f) = |Z'(f)|^2 / (|F(f)|^2 + |Z'(f)|^2 + eps)
        # Maps 0 energy -> 0.0 weight (instead of sigmoid(0) = 0.5) while remaining bounded in [0, 1)
        energy_z = zp_re ** 2 + zp_im ** 2
        energy_f = torch.sum(f_re ** 2 + f_im ** 2, dim=-1, keepdim=True) + 1e-8
        s_weight = energy_z / (energy_f + energy_z + 1e-8)

        # Pulse signal filtering: F' = Norm(F * S)
        gated_re = f_re * s_weight
        gated_im = f_im * s_weight

        norm_factor = torch.norm(gated_re, dim=1, keepdim=True) + 1e-8
        gated_re = gated_re / norm_factor
        gated_im = gated_im / norm_factor

        spec_1d_re = self.channel_fusion(gated_re)  # (B, M, 1)
        spec_1d_im = self.channel_fusion(gated_im)  # (B, M, 1)

        psd = spec_1d_re.squeeze(-1) ** 2 + spec_1d_im.squeeze(-1) ** 2
        return spec_1d_re, spec_1d_im, psd


class BeatFormer(nn.Module):
    """Full BeatFormer network combining CZT, ZOCA, ESD, and ICZT."""

    def __init__(
        self,
        window_len: int = 150,
        num_bins: int = 128,
        fs: float = 30.0,
        low_hz: float = 0.7,
        high_hz: float = 3.0,
        hidden_dim: int = 48,
    ):
        super().__init__()
        self.window_len = window_len
        self.num_bins = num_bins
        self.fs = fs
        self.low_hz = low_hz
        self.high_hz = high_hz

        self.czt = CZTLayer(length=window_len, num_bins=num_bins, fs=fs, low_hz=low_hz, high_hz=high_hz)
        self.zoca = ZOCABlock(num_bins=num_bins, channels=3)
        self.esd = ESDBlock(num_bins=num_bins, channels=3, hidden_dim=hidden_dim)
        self.iczt = ICZTLayer(length=window_len, num_bins=num_bins, fs=fs, low_hz=low_hz, high_hz=high_hz)

    def forward(self, rgb_window: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for a batch of 150-frame RGB traces."""
        if rgb_window.abs().mean() > 1.0:
            mean_c = rgb_window.mean(dim=1, keepdim=True) + 1e-8
            x = (rgb_window / mean_c) - 1.0
        else:
            x = rgb_window

        f_re, f_im = self.czt(x)
        z_re, z_im = self.zoca(f_re, f_im)
        spec_re, spec_im, psd = self.esd(z_re, z_im, f_re, f_im)
        bvp_wave = self.iczt(spec_re, spec_im)

        ortho_loss = (
            self.zoca.proj_q.orthonormal_loss() +
            self.zoca.proj_k.orthonormal_loss() +
            self.zoca.proj_v.orthonormal_loss()
        )

        return {
            "bvp_wave": bvp_wave,
            "psd": psd,
            "freq_axis": self.czt.freq_axis,
            "ortho_loss": ortho_loss,
        }

    def estimate_hr_from_trace(self, rgb_trace: np.ndarray, stride: int = 30) -> Tuple[float, np.ndarray, np.ndarray]:
        """Inference method for arbitrary-length RGB traces using overlapping windows."""
        self.eval()
        mean_c = np.mean(rgb_trace, axis=0, keepdims=True) + 1e-8
        trace_norm = (rgb_trace / mean_c) - 1.0

        t_len = len(trace_norm)
        if t_len < self.window_len:
            pad_len = self.window_len - t_len
            rgb_padded = np.pad(trace_norm, ((0, pad_len), (0, 0)), mode="edge")
        else:
            rgb_padded = trace_norm

        total_padded = len(rgb_padded)
        starts = list(range(0, total_padded - self.window_len + 1, stride))
        if not starts:
            starts = [0]

        bvp_acc = np.zeros(total_padded, dtype=np.float64)
        weight_acc = np.zeros(total_padded, dtype=np.float64)
        hanning = np.hanning(self.window_len)
        psd_list = []

        with torch.no_grad():
            for s in starts:
                win = rgb_padded[s : s + self.window_len]
                win_tensor = torch.from_numpy(win.astype(np.float32)).unsqueeze(0)
                out = self.forward(win_tensor)
                bvp_win = out["bvp_wave"].squeeze().cpu().numpy()
                psd_win = out["psd"].squeeze().cpu().numpy()
                psd_list.append(psd_win)

                bvp_acc[s : s + self.window_len] += bvp_win * hanning
                weight_acc[s : s + self.window_len] += hanning

        valid = weight_acc > 1e-6
        bvp_acc[valid] /= weight_acc[valid]
        bvp_full = bvp_acc[:t_len]

        psd_avg = np.mean(np.array(psd_list), axis=0)
        freqs = self.czt.freq_axis.cpu().numpy()
        peak_idx = int(np.argmax(psd_avg))
        hr_bpm = float(freqs[peak_idx] * 60.0)

        return hr_bpm, bvp_full, psd_avg

    def estimate_hr(self, rgb_trace: np.ndarray, stride: int = 30) -> float:
        """Estimate single global heart rate in BPM from RGB trace."""
        hr_bpm, _, _ = self.estimate_hr_from_trace(rgb_trace, stride=stride)
        return hr_bpm
