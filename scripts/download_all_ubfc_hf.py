"""Automated Sequential Downloader & Transcoder for UBFC-rPPG from HuggingFace.

Downloads subjects from WeiQian98/UBFC-rPPG, transcodes uncompressed raw AVI (1.5 GB)
to lightweight H.264 MP4 (~5 MB, 99.6% size reduction), and cleans up raw AVI immediately
to conserve disk space.
"""

import os
import sys
import glob
import time
import argparse
import subprocess
from typing import List


ALL_SUBJECTS = [
    "subject1", "subject3", "subject4", "subject5", "subject8", "subject9",
    "subject10", "subject11", "subject12", "subject13", "subject14", "subject15",
    "subject16", "subject17", "subject18", "subject20", "subject22", "subject23",
    "subject24", "subject25", "subject26", "subject27", "subject30", "subject31",
    "subject32", "subject33", "subject34", "subject35", "subject36", "subject37",
    "subject38", "subject39", "subject40", "subject41", "subject42", "subject43",
    "subject44", "subject45", "subject46", "subject47", "subject48", "subject49"
]


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


def process_subject(subj: str, out_root: str, hf_token: str) -> bool:
    """Download, transcode, and clean up a single subject."""
    out_dir = os.path.join(out_root, subj)
    os.makedirs(out_dir, exist_ok=True)

    mp4_path = os.path.join(out_dir, "vid.mp4")
    avi_path = os.path.join(out_dir, "vid.avi")
    gt_path = os.path.join(out_dir, "ground_truth.txt")

    # 1. Ensure ground_truth.txt exists
    if not os.path.exists(gt_path) or os.path.getsize(gt_path) == 0:
        gt_url = f"https://huggingface.co/datasets/WeiQian98/UBFC-rPPG/resolve/main/{subj}/ground_truth.txt"
        cmd_gt = ["curl", "-s", "-L", "-H", f"Authorization: Bearer {hf_token}", gt_url, "-o", gt_path]
        subprocess.run(cmd_gt, check=True)

    # 2. Check if video is already transcoded
    if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 100_000:
        print(f"[{subj}] Already completed (MP4 size: {os.path.getsize(mp4_path)/1024/1024:.2f} MB). Skipping.")
        return True

    # 3. Download raw AVI
    avi_url = f"https://huggingface.co/datasets/WeiQian98/UBFC-rPPG/resolve/main/{subj}/vid.avi"
    print(f"[{subj}] Downloading raw video from Hugging Face...")
    t0 = time.time()
    cmd_avi = ["curl", "-L", "-H", f"Authorization: Bearer {hf_token}", avi_url, "-o", avi_path]
    res = subprocess.run(cmd_avi)
    if res.returncode != 0 or not os.path.exists(avi_path) or os.path.getsize(avi_path) < 1_000_000:
        print(f"[{subj}] Error: Download failed or file too small.")
        if os.path.exists(avi_path):
            os.remove(avi_path)
        return False
    dl_time = time.time() - t0
    raw_size_mb = os.path.getsize(avi_path) / 1024 / 1024
    print(f"[{subj}] Downloaded {raw_size_mb:.1f} MB in {dl_time:.1f}s ({raw_size_mb/dl_time:.1f} MB/s).")

    # 4. Transcode immediately to MP4
    print(f"[{subj}] Transcoding raw AVI -> H.264 MP4...")
    t1 = time.time()
    cmd_ff = [
        "ffmpeg", "-y", "-i", avi_path,
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        mp4_path
    ]
    ff_res = subprocess.run(cmd_ff, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ff_res.returncode != 0 or not os.path.exists(mp4_path):
        print(f"[{subj}] Error: Transcoding failed.")
        return False
    tc_time = time.time() - t1
    mp4_size_mb = os.path.getsize(mp4_path) / 1024 / 1024
    print(f"[{subj}] Transcoded to {mp4_size_mb:.2f} MB in {tc_time:.1f}s (Compression: {100*(1 - mp4_size_mb/raw_size_mb):.1f}%).")

    # 5. Delete raw AVI immediately
    os.remove(avi_path)
    print(f"[{subj}] Cleaned up raw AVI. Successfully saved {mp4_path}.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sequentially download and transcode UBFC from HuggingFace")
    parser.add_argument("--out_dir", default="datasets/ubfc", help="UBFC dataset directory")
    parser.add_argument("--max_subjects", type=int, default=None, help="Maximum number of subjects to download")
    parser.add_argument("--subjects", nargs="+", default=None, help="Specific subjects to download")
    args = parser.parse_args()

    hf_token = get_hf_token()
    print("=" * 70)
    print("UBFC-rPPG HUGGING FACE SEQUENTIAL DOWNLOADER & TRANSCODER")
    print("Source: https://huggingface.co/datasets/WeiQian98/UBFC-rPPG")
    print("=" * 70)

    target_subjects = args.subjects if args.subjects else ALL_SUBJECTS
    if args.max_subjects:
        target_subjects = target_subjects[:args.max_subjects]

    print(f"Targeting {len(target_subjects)} subjects in '{args.out_dir}'...")

    successes = 0
    for idx, subj in enumerate(target_subjects, 1):
        print(f"\n---> Processing [{idx}/{len(target_subjects)}]: {subj}")
        ok = process_subject(subj, args.out_dir, hf_token)
        if ok:
            successes += 1
        else:
            print(f"Failed to process {subj}. Continuing to next...")

    print("\n" + "=" * 70)
    print(f"COMPLETED: {successes} / {len(target_subjects)} subjects ready on disk.")
    print("=" * 70)


if __name__ == "__main__":
    main()
