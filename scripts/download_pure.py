#!/usr/bin/env python3
"""Download and extract real PURE dataset files from Hugging Face repository Thinhnb29/PURE.

Repo: https://huggingface.co/datasets/Thinhnb29/PURE
"""

import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path


HF_BASE_URL = "https://huggingface.co/datasets/Thinhnb29/PURE/resolve/main/PURE"


def download_and_extract_session(session_id: str, dest_dir: Path):
    """Download session zip from Hugging Face, extract to dest_dir, and remove zip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_dir = dest_dir / session_id
    target_json = dest_dir / f"{session_id}.json"

    if target_dir.exists() and target_json.exists():
        print(f"[*] Session {session_id} already exists in {dest_dir}. Skipping download.")
        return

    zip_url = f"{HF_BASE_URL}/{session_id}.zip"
    zip_path = dest_dir / f"{session_id}.zip"

    print(f"[*] Downloading {session_id} from {zip_url}...")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, int(count * block_size * 100 / total_size))
            mb_done = (count * block_size) / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r    [{percent}%] {mb_done:.1f}/{mb_total:.1f} MB")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(zip_url, zip_path, reporthook=reporthook)
        print(f"\n[+] Downloaded {session_id}.zip ({zip_path.stat().st_size / 1e6:.1f} MB)")

        print(f"[*] Extracting {session_id}.zip to {dest_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)

        print(f"[+] Extracted {session_id}. Removing zip file...")
        zip_path.unlink()

    except Exception as e:
        print(f"\n[!] Failed to download/extract {session_id}: {e}")
        if zip_path.exists():
            zip_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Download PURE dataset sessions from HuggingFace")
    parser.add_argument("--dest_dir", type=str, default="data/PURE", help="Target destination directory")
    parser.add_argument("--sessions", nargs="+", default=["01-01", "01-02", "01-03", "01-04", "01-05", "01-06"],
                        help="List of sessions to download (e.g. 01-01 01-02)")
    args = parser.parse_args()

    dest = Path(args.dest_dir)
    print(f"Target destination: {dest.resolve()}")
    print(f"Sessions to download: {args.sessions}")

    for s in args.sessions:
        download_and_extract_session(s, dest)

    print("\n[✓] All requested PURE sessions ready in", dest)


if __name__ == "__main__":
    main()
