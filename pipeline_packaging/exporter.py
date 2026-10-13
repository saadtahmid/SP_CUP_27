"""Submission Exporter Module (§1.2 & §4).

Packages the complete official competition submission bundle into:
`/Round1_Baseline/`
  ├── Round1_Baseline_None.mat
  ├── process_round1.m
  ├── target_video.mp4
  ├── reference_signal.csv
  └── processed_output.csv
"""

import os
import shutil
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import cv2

from pipeline_packaging.mat_packager import create_submission_mat_file, verify_submission_mat_file


MATLAB_PROCESS_SCRIPT_TEMPLATE = """% =========================================================================
% 2027 IEEE Signal Processing Cup (FaceVitals) - Round 1 Processing Script
% Script: process_round1.m
% Implements the end-to-end baseline HR extraction pipeline:
% 1. Video reading (target_video.mp4)
% 2. Facial ROI identification and RGB trace extraction
% 3. Butterworth bandpass filtering [0.7 Hz, 3.0 Hz] (42-180 bpm)
% 4. Chirp-Z Transform (CZT) / Spectral HR Peak Estimation
% 5. Exports processed_output.csv with estimated HR and timestamp
% =========================================================================

clear; clc;

video_filename = 'target_video.mp4';
output_csv_filename = 'processed_output.csv';

if ~isfile(video_filename)
    error('Target video %s not found in current directory.', video_filename);
end

fprintf('Loading target video: %s\\n', video_filename);
v = VideoReader(video_filename);
fps = v.FrameRate;
n_frames = floor(v.Duration * fps);

fprintf('Video FPS: %.2f, Total Frames: %d\\n', fps, n_frames);

% Initialize RGB trace arrays
rgb_traces = zeros(n_frames, 3);
frame_idx = 1;

while hasFrame(v) && frame_idx <= n_frames
    frame = readFrame(v);
    [h, w, ~] = size(frame);
    
    % ROI Extraction: Forehead (upper 35% center region) and Cheeks
    fh_y1 = max(1, round(h * 0.20));
    fh_y2 = min(h, round(h * 0.40));
    fh_x1 = max(1, round(w * 0.35));
    fh_x2 = min(w, round(w * 0.65));
    
    roi_pixels = frame(fh_y1:fh_y2, fh_x1:fh_x2, :);
    
    % Compute spatial average per channel (R, G, B)
    mean_r = mean(mean(double(roi_pixels(:,:,1))));
    mean_g = mean(mean(double(roi_pixels(:,:,2))));
    mean_b = mean(mean(double(roi_pixels(:,:,3))));
    
    rgb_traces(frame_idx, :) = [mean_r, mean_g, mean_b];
    frame_idx = frame_idx + 1;
end

rgb_traces = rgb_traces(1:frame_idx-1, :);
N = size(rgb_traces, 1);
timestamps = (0:N-1)' / fps;

fprintf('Extracted RGB traces for %d frames. Applying Plane-Orthogonal-to-Skin (POS)...\\n', N);

% Temporal Normalization
means = mean(rgb_traces, 1);
c_norm = rgb_traces ./ (means + 1e-12);

% POS Projection: S1 = G - B; S2 = G + B - 2*R
s1 = c_norm(:, 2) - c_norm(:, 3);
s2 = c_norm(:, 2) + c_norm(:, 3) - 2.0 * c_norm(:, 1);
alpha = std(s1) / (std(s2) + 1e-12);
h_sig = s1 + alpha * s2;

% Bandpass filter: 0.7 Hz to 3.0 Hz (42 to 180 bpm)
[b, a] = butter(4, [0.7 3.0] / (fps / 2), 'bandpass');
bvp_signal = filtfilt(b, a, h_sig);

% Chirp-Z Transform (czt) HR Estimation
low_hz = 0.7;
high_hz = 3.0;
num_bins = 512;
w_step = (high_hz - low_hz) / (num_bins - 1);

A = exp(1j * 2 * pi * low_hz / fps);
W = exp(-1j * 2 * pi * w_step / fps);

czt_vals = czt(bvp_signal .* hanning(N), num_bins, W, A);
power_spec = abs(czt_vals).^2;

freq_bins = linspace(low_hz, high_hz, num_bins);
bpm_bins = freq_bins * 60.0;

[~, max_idx] = max(power_spec);
estimated_hr = bpm_bins(max_idx);

fprintf('Estimated Heart Rate: %.2f BPM\\n', estimated_hr);

% Save output to processed_output.csv
estimated_hr_series = repmat(estimated_hr, N, 1);
T = table(timestamps, estimated_hr_series, bvp_signal, ...
    'VariableNames', {'timestamp_sec', 'estimated_hr_bpm', 'bvp_signal'});
writetable(T, output_csv_filename);

fprintf('Processed output saved successfully to %s\\n', output_csv_filename);
"""


def export_submission_bundle(
    output_dir: str,
    video_path: str,
    gt_csv_path: str,
    facial_frames: np.ndarray,
    roi_masks: np.ndarray,
    reference_hr_series: np.ndarray,
    estimated_hr_series: np.ndarray,
    metrics: Dict[str, float],
    params: Dict[str, Any],
) -> Dict[str, str]:
    """Export complete competition submission folder matching Section 4 layout."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Round1_Baseline_None.mat
    mat_path = os.path.join(output_dir, "Round1_Baseline_None.mat")
    create_submission_mat_file(
        output_mat_path=mat_path,
        facial_video=facial_frames,
        roi_data=roi_masks,
        reference_signal=reference_hr_series,
        processed_signal=estimated_hr_series,
        metrics=metrics,
        params=params,
    )
    verify_submission_mat_file(mat_path)

    # 2. process_round1.m
    m_script_path = os.path.join(output_dir, "process_round1.m")
    with open(m_script_path, "w") as f:
        f.write(MATLAB_PROCESS_SCRIPT_TEMPLATE)

    # 3. target_video.mp4
    target_vid_dest = os.path.join(output_dir, "target_video.mp4")
    shutil.copyfile(video_path, target_vid_dest)

    # 4. reference_signal.csv
    ref_csv_dest = os.path.join(output_dir, "reference_signal.csv")
    shutil.copyfile(gt_csv_path, ref_csv_dest)

    # 5. processed_output.csv
    proc_csv_dest = os.path.join(output_dir, "processed_output.csv")
    timestamps = np.arange(len(estimated_hr_series)) / float(params.get("fps", 30.0))
    df_out = pd.DataFrame({
        "timestamp_sec": np.round(timestamps, 4),
        "estimated_hr_bpm": np.round(estimated_hr_series, 2),
        "bvp_signal": np.round(estimated_hr_series, 5),
    })
    df_out.to_csv(proc_csv_dest, index=False)

    return {
        "submission_dir": output_dir,
        "mat_file": mat_path,
        "m_script": m_script_path,
        "video_file": target_vid_dest,
        "ref_csv": ref_csv_dest,
        "proc_csv": proc_csv_dest,
        "status": "COMPLETE",
    }
