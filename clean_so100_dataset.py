#!/usr/bin/env python
"""
Clean the *initial* static frames out of a LeRobotDataset that lives on the
Hugging Face Hub.  Frames are discarded **only until** the first significant
motion is detected in either the RGB image *or* the robot joint state – after
that point **all** remaining frames are kept.

The script keeps a small buffer of frames *before* the detected motion so that
context isn’t lost (e.g. the object just before it starts to move).

Typical usage
-------------
$ python clean_so100_dataset.py \
    --source-repo jchun/so100_cleaning_20250524_161038 \
    --target-repo YOUR_USERNAME/so100_cleaning_trimmed \
    --image-thresh 5.0 --joint-thresh 1e-3 --pre-buffer 5

Dependencies
------------
* datasets ≥ 2.18.0
* pillow
* numpy
* tqdm (progress‑bar)
* huggingface‑hub (for push)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np  # <class 'numpy.ndarray'> from numpy/__init__.py
from datasets import Dataset, load_dataset  # Dataset class from datasets/arrow_dataset.py
from huggingface_hub import HfApi  # HfApi class from huggingface_hub/hf_api.py
from PIL import Image  # Image class from PIL/Image.py
from tqdm import tqdm  # tqdm progress‑bar from tqdm/std.py

###############################################################################
# Utility functions                                                           #
###############################################################################

def img_mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    """Return mean absolute pixel difference between two RGB PIL images."""
    # Convert to NumPy arrays of int16 so subtraction won’t underflow.
    arr_a: np.ndarray = np.asarray(a, dtype=np.int16)  # shape (H,W,3)
    arr_b: np.ndarray = np.asarray(b, dtype=np.int16)
    return float(np.mean(np.abs(arr_a - arr_b)))


def detect_first_motion(frames: List[dict], image_thresh: float, joint_thresh: float, pre_buffer: int) -> int:
    """Return the frame index *within the episode* where motion begins."""
    prev_img: Image.Image | None = None
    prev_state: np.ndarray | None = None

    for idx, row in enumerate(frames):
        img: Image.Image = row["obs"]["rgb_front"]  # PIL.Image.Image
        state: np.ndarray = np.asarray(row["obs"]["robot_state"], dtype=np.float32)

        if prev_img is None:
            prev_img, prev_state = img, state
            continue

        img_delta: float = img_mean_abs_diff(img, prev_img)
        joint_delta: float = float(np.linalg.norm(state - prev_state, ord=1))

        if (img_delta > image_thresh) or (joint_delta > joint_thresh):
            # Found first meaningful change.
            return max(0, idx - pre_buffer)

        prev_img, prev_state = img, state

    # No motion detected – keep everything.
    return 0


def trim_episode(frames: List[dict], start_index: int) -> List[dict]:
    """Cut the frames *before* `start_index` inside a single episode."""
    return [f for f in frames if f["frame_index"] >= start_index]

###############################################################################
# Main routine                                                                #
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Trim initial static frames from a LeRobotDataset.")
    parser.add_argument("--source-repo", required=True, help="HF repo id of the source dataset")
    parser.add_argument("--target-repo", required=True, help="HF repo id where the cleaned dataset will be pushed")
    parser.add_argument("--split", default="train", help="Dataset split to process (default: train)")
    parser.add_argument("--image-thresh", type=float, default=5.0, help="Mean‑ABS pixel diff threshold (uint8 scale)")
    parser.add_argument("--joint-thresh", type=float, default=1e-3, help="L1 diff threshold for robot_state vector")
    parser.add_argument("--pre-buffer", type=int, default=5, help="Frames kept *before* first motion")
    parser.add_argument("--local-out", type=Path, default=Path("./trimmed_dataset"), help="Local output directory")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    print(f"⏬ Downloading {args.source_repo} …")
    ds = load_dataset(args.source_repo, split=args.split, streaming=False)  # Dataset object
    # Ensure frames are ordered (episode_index, frame_index)
    ds = ds.sort(["episode_index", "frame_index"])  # does an Arrow sort under the hood

    # ---------------------------------------------------------------------
    # Group frames by episode in memory (OK for ≤ few 100k frames)
    episodes: defaultdict[int, List[dict]] = defaultdict(list)
    for row in tqdm(ds, desc="Buffering episodes"):
        episodes[row["episode_index"]].append(row)

    # ---------------------------------------------------------------------
    cleaned_frames: List[dict] = []
    for ep_id, frames in tqdm(episodes.items(), desc="Trimming episodes"):
        cut_at = detect_first_motion(frames, args.image_thresh, args.joint_thresh, args.pre_buffer)
        cleaned_frames.extend(trim_episode(frames, cut_at))

    # ---------------------------------------------------------------------
    print("📦 Re‑assembling cleaned dataset …")
    cleaned_ds: Dataset = Dataset.from_list(cleaned_frames)  # new Dataset object in memory

    print(f"💾 Saving to {args.local_out}")
    cleaned_ds.save_to_disk(str(args.local_out))

    # ---------------------------------------------------------------------
    print(f"🚀 Pushing to HF Hub → {args.target_repo}")
    api = HfApi()
    api.create_repo(args.target_repo, repo_type="dataset", exist_ok=True)
    cleaned_ds.push_to_hub(args.target_repo)
    print("✅ Done – cleaned dataset pushed!")


if __name__ == "__main__":
    main()
