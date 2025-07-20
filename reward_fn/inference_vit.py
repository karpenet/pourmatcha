#!/usr/bin/env python3
"""
Inference script for task completion estimation using trained ViT model
"""

import argparse
import collections
import time

import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from train_completion_vit import ProgressEstimator


def predict_episode_sequence(
    estimator: ProgressEstimator, episode_dir: str
):
    """Predict completion for all frames in an episode"""

    episode_path = Path(episode_dir)
    frames_dir = episode_path / "frames"

    if not frames_dir.exists():
        raise ValueError(f"Frames directory not found: {frames_dir}")

    # Get all frame files and sort them
    frame_files = sorted([f for f in frames_dir.glob("frame_*.jpg")])

    predictions = []
    frame_indices = []

    print(f"Processing {len(frame_files)} frames...")

    estimator.reset()

    for frame_file in frame_files:
        # Extract frame index from filename
        frame_idx = int(frame_file.stem.split('_')[1])

        image = Image.open(frame_file).convert('RGB')
        completion = estimator.predict(image)

        predictions.append(completion)
        frame_indices.append(frame_idx)

    return np.array(predictions), np.array(frame_indices)


def visualize_episode_predictions(
    predictions, frame_indices, output_path: str = None
):
    """Create visualization of episode predictions"""

    # Create ground truth completion (linear progression)
    max_frame = frame_indices.max()
    true_completions = frame_indices / max_frame

    plt.figure(figsize=(10, 6))

    plt.plot(frame_indices, true_completions, 'b-',
             label='Ground Truth (Linear)', linewidth=2)
    plt.plot(frame_indices, predictions, 'r-',
             label='Model Prediction', linewidth=2, alpha=0.8)

    plt.xlabel('Frame Index')
    plt.ylabel('Task Completion')
    plt.title('Task Completion Estimation Over Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # Add statistics
    mae = np.mean(np.abs(predictions - true_completions))
    plt.text(0.02, 0.98, f'MAE: {mae:.4f}', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")

    plt.show()


def run_live_inference(
    estimator: ProgressEstimator, camera_index=0, history_size=200
):
    """Run live inference from camera and plot completion."""

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return

    # --- Matplotlib setup for vertically stacked view ---
    plt.ion()
    fig = plt.figure(figsize=(12, 8))

    # Create subplot grid with larger top image
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.01)

    # Image subplot (larger, on top)
    ax_img = fig.add_subplot(gs[0])
    ax_img.set_title("Camera Feed")
    ax_img.axis('off')

    # Plot subplot (smaller, below)
    ax_plot = fig.add_subplot(gs[1])
    completions = collections.deque(maxlen=history_size)
    timestamps = collections.deque(maxlen=history_size)
    start_time = time.time()
    ax_plot.set_ylim(-0.05, 1.05)
    ax_plot.set_xlabel("Time (s)")
    ax_plot.set_ylabel("Completion")
    ax_plot.grid(True, alpha=0.3)
    line, = ax_plot.plot([], [], 'g-', linewidth=2)  # Initial empty line

    # Use a placeholder for the first frame display
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture initial frame.")
        cap.release()
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_display = ax_img.imshow(frame_rgb)

    print("Starting live inference. Close the plot window to quit.")

    estimator.reset()

    while plt.fignum_exists(fig.number):  # Loop until window is closed
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Predict completion
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        completion = estimator.predict(image)

        # Update data
        completions.append(completion)
        timestamps.append(time.time() - start_time)

        # --- Update Display ---
        # Update image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_display.set_data(frame_rgb)
        ax_img.set_title(f"Task Completion: {completion:.2%}")

        # Update plot
        line.set_data(list(timestamps), list(completions))
        ax_plot.relim()
        ax_plot.autoscale_view(scalex=True, scaley=False)

        # Redraw
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Cleanup
    cap.release()
    plt.ioff()
    print("Live inference stopped.")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Inference with trained ViT model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--image", type=str, help="Path to single image for prediction"
    )
    parser.add_argument(
        "--episode",
        type=str,
        help="Path to episode directory for sequence prediction",
    )
    parser.add_argument(
        "--output", type=str, help="Output path for visualization"
    )
    parser.add_argument(
        "--live", action="store_true", help="Run live inference from camera feed"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index of the camera to use for live inference.",
    )

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config from checkpoint to get image_size
    print(f"Loading config from {args.checkpoint}")

    # Initialize ProgressEstimator
    print("Initializing ProgressEstimator...")
    estimator = ProgressEstimator(
        model_path=args.checkpoint,
        device=device
    )

    if args.live:
        run_live_inference(
            estimator, camera_index=args.camera_index
        )

    elif args.image:
        # Single image prediction
        print(f"\nPredicting completion for single image: {args.image}")
        image = Image.open(args.image).convert('RGB')
        completion = estimator.predict(image)

        print(
            f"Predicted task completion: {completion:.4f} "
            f"({completion*100:.1f}%)"
        )

        # Show image with prediction
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f'Task Completion: {completion:.1%}')
        plt.axis('off')

        if args.output:
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"Result saved to {args.output}")

        plt.show()

    elif args.episode:
        # Episode sequence prediction
        print(f"\nPredicting completion for episode: {args.episode}")
        predictions, frame_indices = predict_episode_sequence(
            estimator, args.episode)

        print(f"Processed {len(predictions)} frames")
        print(
            f"Completion range: {predictions.min():.4f} to "
            f"{predictions.max():.4f}"
        )
        print(
            f"Final completion: {predictions[-1]:.4f} "
            f"({predictions[-1]*100:.1f}%)"
        )

        # Visualize sequence
        output_path = args.output if args.output else "episode_predictions.png"
        visualize_episode_predictions(predictions, frame_indices, output_path)

    else:
        print("Please specify --live, --image, or --episode for inference.")


if __name__ == "__main__":
    main()
