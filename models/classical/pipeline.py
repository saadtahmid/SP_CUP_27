"""Pipeline v1: Unified Classical rPPG Engine with CZT Spectral Estimation.

Integrates POS, CHROM, GREEN, and ICA algorithms with:
1. Temporal channel normalization
2. 4th-order zero-phase Butterworth bandpass filter [0.7 Hz, 3.0 Hz] (42-180 bpm)
3. High-resolution Chirp-Z Transform (CZT) HR peak estimator (vs FFT)
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np
from scipy import signal

from models.classical.pos import pos_process
from models.classical.chrom import chrom_process
from models.classical.green_ica import green_process, ica_process
from estimation.czt_estimator import CZTHRTransformer
from estimation.fft_estimator import estimate_hr_welch


class ClassicalRPPGPipeline:
    """End-to-end Classical rPPG Pipeline."""

    SUPPORTED_METHODS = ["POS", "CHROM", "GREEN", "ICA"]

    def __init__(
        self,
        method: str = "POS",
        fs: float = 30.0,
        low_hz: float = 0.7,
        high_hz: float = 3.0,
        use_czt: bool = True,
    ):
        method_upper = method.upper()
        if method_upper not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method '{method}' not supported. Choose from {self.SUPPORTED_METHODS}")

        self.method = method_upper
        self.fs = float(fs)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.use_czt = bool(use_czt)

        self.czt_estimator = CZTHRTransformer(fs=self.fs, low_hz=self.low_hz, high_hz=self.high_hz)

    def extract_bvp(self, rgb_signals: np.ndarray) -> np.ndarray:
        """Extract BVP waveform from raw RGB mean traces."""
        if self.method == "POS":
            return pos_process(rgb_signals, fs=self.fs)
        elif self.method == "CHROM":
            return chrom_process(rgb_signals, fs=self.fs)
        elif self.method == "GREEN":
            return green_process(rgb_signals, fs=self.fs)
        elif self.method == "ICA":
            return ica_process(rgb_signals, fs=self.fs)
        else:
            return pos_process(rgb_signals, fs=self.fs)

    def process(
        self,
        rgb_signals: np.ndarray,
        return_spectrum: bool = False,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Run complete extraction and HR estimation."""
        bvp = self.extract_bvp(rgb_signals)

        if self.use_czt:
            if return_spectrum:
                hr_bpm, freqs, power = self.czt_estimator.estimate_hr(bvp, return_spectrum=True)
            else:
                hr_bpm = self.czt_estimator.estimate_hr(bvp, return_spectrum=False)
                freqs, power = None, None
        else:
            if return_spectrum:
                hr_bpm, freqs, power = estimate_hr_welch(
                    bvp, fs=self.fs, low_hz=self.low_hz, high_hz=self.high_hz, return_spectrum=True
                )
            else:
                hr_bpm = estimate_hr_welch(
                    bvp, fs=self.fs, low_hz=self.low_hz, high_hz=self.high_hz, return_spectrum=False
                )
                freqs, power = None, None

        result = {
            "method": self.method,
            "bvp": bvp,
            "hr_bpm": float(hr_bpm),
            "fs": self.fs,
            "use_czt": self.use_czt,
        }
        if return_spectrum:
            result["spectrum_freqs"] = freqs
            result["spectrum_power"] = power

        return result
