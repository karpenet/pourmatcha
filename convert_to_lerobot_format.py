#!/usr/bin/env python3
"""
Convert custom folding datasets to LeRobot HuggingFace format.

This script converts data from reward_fn/fold_data/ and 
reward_fn/fold_data_negative/ to HuggingFace LeRobot format for training and 
sharing.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count
import numpy as np
from PIL import Image
import logging

# Import LeRobot components
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_collection_metadata(data_dir: Path) -> Dict:
    """Load the collection metadata JSON file."""
    metadata_path = data_dir / "collection_metadata.json"
    with open(metadata_path, 'r') as f:
        return json.load(f)


def get_sorted_episode_dirs(data_dir: Path) -> List[Path]:
    """Get all episode directories sorted by episode number."""
    episode_dirs = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.startswith('episode_'):
            episode_dirs.append(item)

    # Sort by episode number
    episode_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    return episode_dirs


def get_sorted_frame_paths(frames_dir: Path) -> List[Path]:
    """Get all frame paths sorted by frame number."""
    frame_paths = []
    for frame_file in frames_dir.glob("frame_*.jpg"):
        frame_paths.append(frame_file)

    # Sort by frame number
    frame_paths.sort(key=lambda x: int(x.stem.split('_')[1]))
    return frame_paths


def process_single_episode(args: Tuple[Path, int, str, int]) -> Tuple[int, List[np.ndarray]]:
    """
    Process a complete episode in parallel.

    Args:
        args: Tuple of (episode_dir, ep_idx, task_description, dataset_fps)

    Returns:
        Tuple of (episode_index, processed_images_list)
    """
    episode_dir, ep_idx, task_description, dataset_fps = args

    # Load episode metadata
    metadata_path = episode_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            episode_meta = json.load(f)
        actual_fps = episode_meta.get('actual_fps', dataset_fps)
    else:
        actual_fps = dataset_fps

    # Get frame paths
    frames_dir = episode_dir / "frames"
    if not frames_dir.exists():
        logger.warning(
            f"No frames directory found for {episode_dir}, skipping")
        return ep_idx, []

    frame_paths = get_sorted_frame_paths(frames_dir)
    num_frames = len(frame_paths)

    if num_frames == 0:
        logger.warning(f"No frames found in {frames_dir}, skipping")
        return ep_idx, []

    logger.info(f"Episode {ep_idx} has {num_frames} frames, "
                f"actual_fps: {actual_fps}")

    # Process all images for this episode
    processed_images = []
    for frame_idx, frame_path in enumerate(frame_paths):
        # Load and process image
        image = Image.open(frame_path)

        # Convert to RGB if needed and resize to expected resolution
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to match so100 config (640x480)
        image = image.resize((640, 480), Image.LANCZOS)

        # Convert to numpy array (H, W, C)
        image_array = np.array(image)
        processed_images.append(image_array)

    logger.info(
        f"Completed processing episode {ep_idx} with {len(processed_images)} frames")
    return ep_idx, processed_images


def convert_dataset(
    source_dir: Path,
    repo_id: str,
    task_description: str,
    output_dir: Path = None,
    push_to_hub: bool = True,
    num_workers: int = None
) -> LeRobotDataset:
    """
    Convert a dataset from custom format to LeRobot format.

    Args:
        source_dir: Path to source data directory (fold_data or 
                   fold_data_negative)
        repo_id: HuggingFace repo ID for the dataset
        task_description: Description of the task being performed
        output_dir: Local directory to save the dataset (optional)
        push_to_hub: Whether to push to HuggingFace Hub
        num_workers: Number of parallel workers for episode processing
    """
    if num_workers is None:
        num_workers = cpu_count()

    logger.info(f"Converting dataset from {source_dir} to LeRobot format...")
    logger.info(f"Using {num_workers} parallel workers for episode processing")

    # Load collection metadata
    collection_meta = load_collection_metadata(source_dir)
    fps = collection_meta.get('fps', 30)

    logger.info(f"Collection metadata: {collection_meta['total_episodes']} "
                f"episodes, {fps} fps")

    # Get episode directories
    episode_dirs = get_sorted_episode_dirs(source_dir)
    logger.info(f"Found {len(episode_dirs)} episodes")

    # Define features - only images since no robot data
    features = {
        # Default LeRobot features
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},

        # Camera observation (main camera from frames)
        "observation.images.main": {
            "dtype": "video",
            "shape": (480, 640, 3),  # Height, Width, Channels
            "names": ["height", "width", "channels"]
        }
    }

    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        robot_type="so100",
        features=features,
        use_videos=True,  # Use videos for efficient storage
        tolerance_s=1e-4
    )

    logger.info(f"Created dataset with repo_id: {repo_id}")

    # Process all episodes in parallel
    logger.info(f"Processing {len(episode_dirs)} episodes in parallel...")

    # Prepare arguments for parallel processing
    args = [(episode_dir, ep_idx, task_description, fps)
            for ep_idx, episode_dir in enumerate(episode_dirs)]

    # Process episodes in parallel
    with Pool(processes=num_workers) as pool:
        episode_results = pool.map(process_single_episode, args)

    # Sort results by episode index to maintain order
    episode_results.sort(key=lambda x: x[0])

    logger.info("All episodes processed. Adding to dataset...")

    # Add all processed episodes to dataset sequentially
    for ep_idx, processed_images in episode_results:
        if not processed_images:  # Skip empty episodes
            continue

        logger.info(
            f"Adding episode {ep_idx} with {len(processed_images)} frames to dataset...")

        # Add frames to dataset sequentially (LeRobot operations not thread-safe)
        for frame_idx, image_array in enumerate(processed_images):
            # Create frame data - only image and required metadata
            # Don't include timestamp - let LeRobot auto-generate it first
            frame_data = {
                "observation.images.main": image_array,
                "task": task_description
            }

            # Add frame to dataset
            dataset.add_frame(frame_data)

            # Override auto-generated timestamp with dataset fps
            # This ensures timestamps sync with LeRobot's expected intervals
            custom_timestamp = frame_idx / fps
            dataset.episode_buffer["timestamp"][-1] = custom_timestamp

        # Save the episode
        logger.info(f"Saving episode {ep_idx}")
        dataset.save_episode()

    # Consolidate dataset (compute statistics, etc.)
    logger.info("Consolidating dataset...")
    dataset.consolidate(run_compute_stats=True)

    # Push to hub if requested
    if push_to_hub:
        logger.info(f"Pushing dataset to HuggingFace Hub: {repo_id}")
        dataset.push_to_hub(tags=["so100", "folding", "custom_data"])

    logger.info("Dataset conversion completed successfully!")
    return dataset


def main():
    """Main function to convert both datasets."""

    # Get current timestamp for unique repo names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define paths
    base_dir = Path("reward_fn")
    fold_data_dir = base_dir / "fold_data"
    fold_data_negative_dir = base_dir / "fold_data_negative"

    # Check if directories exist
    if not fold_data_dir.exists():
        raise FileNotFoundError(f"Directory {fold_data_dir} not found")
    if not fold_data_negative_dir.exists():
        raise FileNotFoundError(
            f"Directory {fold_data_negative_dir} not found")

    # Determine number of workers to use
    num_workers = cpu_count()
    logger.info(f"System has {num_workers} CPU cores available")

    # Convert positive folding dataset
    logger.info("=" * 60)
    logger.info("Converting positive folding dataset...")
    logger.info("=" * 60)

    positive_repo_id = f"jchun/so100_folding_positive_{timestamp}"
    convert_dataset(
        source_dir=fold_data_dir,
        repo_id=positive_repo_id,
        task_description="Fold the cloth correctly",
        push_to_hub=True,
        num_workers=num_workers
    )

    # Convert negative folding dataset
    logger.info("=" * 60)
    logger.info("Converting negative folding dataset...")
    logger.info("=" * 60)

    negative_repo_id = f"jchun/so100_folding_negative_{timestamp}"
    convert_dataset(
        source_dir=fold_data_negative_dir,
        repo_id=negative_repo_id,
        task_description="Incorrect cloth folding demonstration",
        push_to_hub=True,
        num_workers=num_workers
    )

    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Positive dataset: {positive_repo_id}")
    logger.info(f"Negative dataset: {negative_repo_id}")
    logger.info("Both datasets have been converted and uploaded to "
                "HuggingFace Hub.")


if __name__ == "__main__":
    main()
