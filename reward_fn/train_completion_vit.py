#!/usr/bin/env python3
"""
Vision Transformer Training Script for Task Completion Estimation
Regresses frame indices to estimate task completion percentage (0-1)
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from PIL import Image
import wandb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
from collections import deque
import cv2
from multiprocessing import Pool
from filelock import FileLock
from torch.cuda.amp import autocast, GradScaler

# Import LeRobot components
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Set random seeds for reproducibility


def set_seed(seed: int = 42):
    """Set seeds for reproducible training"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TaskCompletionDataset(Dataset):
    """Dataset for task completion estimation from LeRobot datasets."""

    def __init__(self, positive_repo_ids: List[str], transform=None,
                 negative_repo_ids: Optional[List[str]] = None,
                 num_frames: int = 1, cache_dir: str = None,
                 completion_threshold: float = 0.9):
        """
        Initialize dataset from LeRobot repositories.

        Args:
            positive_repo_ids: List of repo IDs with positive completion examples
            negative_repo_ids: List of repo IDs with negative/failed examples  
            transform: Image transforms to apply
            num_frames: Number of frames to use for temporal models
            cache_dir: Directory for caching optical flow computations
            completion_threshold: Threshold above which frames are labeled as complete
        """
        self.positive_repo_ids = positive_repo_ids
        self.negative_repo_ids = negative_repo_ids or []
        self.transform = transform
        self.num_frames = num_frames
        self.completion_threshold = completion_threshold
        self.samples = []

        # Load positive datasets
        print("Loading positive datasets...")
        for repo_id in tqdm(positive_repo_ids, desc="Loading positive repos"):
            try:
                dataset = LeRobotDataset(repo_id, root=cache_dir)
                self._process_dataset(dataset, is_positive=True)
            except Exception as e:
                print(f"Warning: Failed to load {repo_id}: {e}")
                continue

        num_pos_samples = len(self.samples)

        # Load negative datasets
        if self.negative_repo_ids:
            print("Loading negative datasets...")
            for repo_id in tqdm(negative_repo_ids, desc="Loading negative repos"):
                try:
                    dataset = LeRobotDataset(repo_id, root=cache_dir)
                    self._process_dataset(dataset, is_positive=False)
                except Exception as e:
                    print(f"Warning: Failed to load {repo_id}: {e}")
                    continue

        num_neg_samples = len(self.samples) - num_pos_samples
        print(f"Loaded {num_pos_samples} positive samples from "
              f"{len(positive_repo_ids)} repos and {num_neg_samples} negative "
              f"samples from {len(self.negative_repo_ids)} repos.")

        # Calculate sample weights for balanced training if using multi-frame
        if self.num_frames > 1:
            print("Calculating sample weights based on completion progression...")
            self._calculate_sample_weights()

    def _process_dataset(self, dataset: LeRobotDataset, is_positive: bool):
        """Process a single LeRobot dataset and extract samples"""
        total_frames = len(dataset)
        if total_frames == 0:
            return

        # Group frames by episode
        episodes = {}
        for frame_idx in range(total_frames):
            episode_idx = dataset.episode_indices[frame_idx]
            if episode_idx not in episodes:
                episodes[episode_idx] = []
            episodes[episode_idx].append(frame_idx)

        # Process each episode
        for episode_idx, frame_indices in episodes.items():
            if len(frame_indices) < 60:  # Skip short episodes
                continue

            episode_length = len(frame_indices)

            for i, global_frame_idx in enumerate(frame_indices):
                if is_positive:
                    # Calculate completion percentage for positive examples
                    completion_pct = i / \
                        (episode_length - 1) if episode_length > 1 else 0.0
                    completion_pct = min(1.0, max(0.0, completion_pct))

                    # Apply thresholding
                    if completion_pct > 0.92:
                        completion_pct = 1.0
                    elif completion_pct < 0.12:
                        completion_pct = 0.0
                else:
                    # All negative examples have 0 completion
                    completion_pct = 0.0

                # Get the actual image from the dataset
                frame_data = dataset[global_frame_idx]
                image = frame_data.get("observation.images.main")

                if image is None:
                    continue

                self.samples.append({
                    'dataset': dataset,
                    'global_frame_idx': global_frame_idx,
                    'episode_idx': episode_idx,
                    'episode_frame_idx': i,
                    'episode_length': episode_length,
                    'completion': completion_pct,
                    'is_positive': is_positive,
                    'frame_indices': frame_indices  # All frames in this episode
                })

    def _calculate_sample_weights(self):
        """Calculate sample weights for balanced training"""
        # For multi-frame models, weight samples by completion progression
        # This helps the model learn from more informative temporal sequences
        weights = []
        for sample in self.samples:
            if sample['is_positive']:
                # Weight positive samples by how much completion changes
                # around this frame (more change = more informative)
                completion = sample['completion']
                # Give higher weight to middle completion values
                weight = 1.0 + 2.0 * completion * (1.0 - completion)
            else:
                # Negative samples get baseline weight
                weight = 0.5

            weights.append(weight)

        self.sample_weights = weights
        print(
            f"Sample weights calculated for {len(self.sample_weights)} samples.")

    def get_sampler(self):
        """Get weighted sampler for balanced training"""
        if hasattr(self, 'sample_weights'):
            return WeightedRandomSampler(
                self.sample_weights, len(self.sample_weights))
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        dataset = sample['dataset']
        episode_frame_idx = sample['episode_frame_idx']
        frame_indices = sample['frame_indices']

        image_sequence = []
        if self.num_frames > 1:
            # Get sequence of frames for temporal models
            start_index = max(0, episode_frame_idx - self.num_frames + 1)
            sequence_indices = list(range(start_index, episode_frame_idx + 1))

            # Pad by repeating first frame if needed
            while len(sequence_indices) < self.num_frames:
                sequence_indices.insert(0, start_index)

            for seq_idx in sequence_indices:
                global_frame_idx = frame_indices[seq_idx]
                frame_data = dataset[global_frame_idx]
                image = frame_data["observation.images.main"]

                # Convert to PIL Image
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                elif hasattr(image, 'numpy'):  # torch tensor
                    image = Image.fromarray(image.numpy())

                image = image.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                image_sequence.append(image)

            images = torch.stack(image_sequence)
        else:
            # Single frame
            frame_data = dataset[sample['global_frame_idx']]
            image = frame_data["observation.images.main"]

            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif hasattr(image, 'numpy'):
                image = Image.fromarray(image.numpy())

            image = image.convert('RGB')
            if self.transform:
                image = self.transform(image)
            images = image

        # Return image(s) and completion target
        return images, torch.tensor(sample['completion'], dtype=torch.float32)


class TaskCompletionViT(nn.Module):
    """Vision Transformer for task completion estimation."""

    def __init__(self, pretrained_model="vit_b_16", dropout_rate=0.4,
                 num_frames=1, aggregator='lstm'):
        super().__init__()
        self.num_frames = num_frames
        self.aggregator = aggregator

        # Load pretrained ViT
        if pretrained_model == "vit_b_16":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            hidden_dim = 768
        else:
            raise ValueError(f"Unsupported model: {pretrained_model}")

        # Remove the classification head
        self.backbone.heads = nn.Identity()

        # Add dropout
        self.dropout = nn.Dropout(dropout_rate)

        if num_frames == 1:
            # Single frame model
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            # Multi-frame temporal model
            if aggregator == 'lstm':
                self.temporal_aggregator = nn.LSTM(
                    hidden_dim, 256, batch_first=True, dropout=dropout_rate)
                temporal_out_dim = 256
            elif aggregator == 'sum':
                self.temporal_aggregator = None
                temporal_out_dim = hidden_dim
            elif aggregator == 'concat':
                self.temporal_aggregator = None
                temporal_out_dim = hidden_dim * num_frames
            else:
                raise ValueError(f"Unknown aggregator: {aggregator}")

            self.regressor = nn.Sequential(
                nn.Linear(temporal_out_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def get_embedding(self, x):
        """Get feature embedding from backbone"""
        return self.backbone(x)

    def forward_sequence(self, embeddings):
        """Forward pass for pre-computed embeddings sequence"""
        B, T, D = embeddings.shape

        if self.aggregator == 'lstm':
            lstm_out, (hidden, cell) = self.temporal_aggregator(embeddings)
            features = hidden[-1]  # Use last hidden state
        elif self.aggregator == 'sum':
            features = torch.sum(embeddings, dim=1)
        elif self.aggregator == 'concat':
            features = embeddings.view(B, -1)

        features = self.dropout(features)
        return self.regressor(features)

    def forward(self, x):
        if self.num_frames == 1:
            # Single frame: x shape is (B, C, H, W)
            features = self.get_embedding(x)
            features = self.dropout(features)
            return self.regressor(features)
        else:
            # Multi-frame: x shape is (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)

            # Get embeddings for all frames
            embeddings = self.get_embedding(x)
            embeddings = embeddings.view(B, T, -1)

            return self.forward_sequence(embeddings)


def create_transforms(image_size, num_frames):
    """Create training and validation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_one_epoch(model, train_loader, optimizer, criterion, device,
                    scaler=None, amp_dtype=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []

    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, target_completion) in enumerate(progress_bar):
        images = images.to(device)
        target_completion = target_completion.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            # Use automatic mixed precision
            with autocast(dtype=amp_dtype, enabled=True):
                outputs = model(images).squeeze()
                loss = criterion(outputs, target_completion)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images).squeeze()
            loss = criterion(outputs, target_completion)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Store predictions and targets for metrics
        predictions.extend(outputs.detach().cpu().numpy())
        targets.extend(target_completion.detach().cpu().numpy())

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
        })

    avg_loss = total_loss / len(train_loader)
    mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))

    return avg_loss, mae, predictions, targets


def validate(model, val_loader, criterion, device, amp_dtype=None):
    """Validate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for images, target_completion in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            target_completion = target_completion.to(device)

            with autocast(dtype=amp_dtype, enabled=amp_dtype is not None):
                outputs = model(images).squeeze()
                loss = criterion(outputs, target_completion)

            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(target_completion.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))

    return avg_loss, mae, predictions, targets


def train_model(model, train_loader, val_loader, num_epochs, device,
                save_dir, model_config, scaler=None, amp_dtype=None):
    """Main training loop"""
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(),
                      lr=model_config['lr'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_mae = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_mae, train_preds, train_targets = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, amp_dtype)

        # Validate
        val_loss, val_mae, val_preds, val_targets = validate(
            model, val_loader, criterion, device, amp_dtype)

        # Step scheduler
        scheduler.step()

        # Logging
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_mae': train_mae,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_mae,
                'config': model_config
            }

            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"New best model saved with MAE: {best_mae:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement")
            break

    print(f"\nTraining completed! Best validation MAE: {best_mae:.4f}")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Train ViT for task completion estimation using LeRobot datasets")
    parser.add_argument("--positive_repo_ids", type=str, nargs='+', required=True,
                        help="HuggingFace repo IDs with positive completion examples")
    parser.add_argument("--negative_repo_ids", type=str, nargs='*', default=None,
                        help="HuggingFace repo IDs with negative/failed examples")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout rate")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--num_frames", type=int, default=1,
                        help="Number of frames to use for estimation")
    parser.add_argument("--aggregator", type=str, default='lstm',
                        choices=['lstm', 'sum', 'concat'],
                        help="Temporal aggregation method")
    parser.add_argument("--model_name", type=str, default="vit_b_16",
                        help="ViT model variant")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="task-completion-vit",
                        help="Wandb project name")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory for caching LeRobot datasets")
    parser.add_argument("--precision", type=str, default='fp32',
                        choices=['fp32', 'fp16', 'bf16'],
                        help="Training precision")
    parser.add_argument("--completion_threshold", type=float, default=0.9,
                        help="Completion threshold for positive examples")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Handle precision
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        print("Warning: BF16 not supported. Falling back to FP32.")
        args.precision = 'fp32'

    amp_dtype = None
    if args.precision == 'bf16':
        amp_dtype = torch.bfloat16
    elif args.precision == 'fp16':
        amp_dtype = torch.float16

    scaler = GradScaler(enabled=(args.precision != 'fp32'))

    # Initialize wandb
    wandb_name = (f"lerobot-vit-{args.model_name}-lr{args.lr}-bs{args.batch_size}-"
                  f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    wandb.init(project=args.wandb_project, config=vars(args), name=wandb_name)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Create transforms
    train_transform, val_transform = create_transforms(
        args.image_size, args.num_frames)

    # Create datasets
    print("Creating training dataset...")
    full_dataset = TaskCompletionDataset(
        positive_repo_ids=args.positive_repo_ids,
        negative_repo_ids=args.negative_repo_ids,
        transform=train_transform,
        num_frames=args.num_frames,
        cache_dir=args.cache_dir,
        completion_threshold=args.completion_threshold
    )

    # Split dataset
    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=args.val_split,
        random_state=args.seed,
        stratify=[s['is_positive'] for s in full_dataset.samples]
    )

    # Create subset datasets
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]

    train_dataset = TaskCompletionDataset.__new__(TaskCompletionDataset)
    train_dataset.__dict__ = full_dataset.__dict__.copy()
    train_dataset.samples = train_samples
    train_dataset.transform = train_transform

    val_dataset = TaskCompletionDataset.__new__(TaskCompletionDataset)
    val_dataset.__dict__ = full_dataset.__dict__.copy()
    val_dataset.samples = val_samples
    val_dataset.transform = val_transform

    # Create data loaders
    train_sampler = train_dataset.get_sampler()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=4, pin_memory=True)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TaskCompletionViT(
        pretrained_model=args.model_name,
        dropout_rate=args.dropout,
        num_frames=args.num_frames,
        aggregator=args.aggregator
    ).to(device)

    model_config = {
        'model_name': args.model_name,
        'dropout': args.dropout,
        'num_frames': args.num_frames,
        'aggregator': args.aggregator,
        'image_size': args.image_size,
        'lr': args.lr,
        'precision': args.precision
    }

    print(f"Model created on {device}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    train_model(model, train_loader, val_loader, args.epochs, device,
                args.save_dir, model_config, scaler, amp_dtype)


# ... ProgressEstimator class remains the same ...

class ProgressEstimator:
    """Estimates task completion progress using a trained ViT model."""

    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model and config from checkpoint
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']

        self.num_frames = config.get('num_frames', 1)
        self.image_size = config.get('image_size', 224)
        self.embedding_cache = deque(maxlen=self.num_frames)

        # Handle precision from loaded model
        self.precision = config.get('precision', 'fp32')
        self.amp_dtype = None
        if self.precision == 'bf16':
            self.amp_dtype = torch.bfloat16
        elif self.precision == 'fp16':
            self.amp_dtype = torch.float16
        self.autocast_enabled = self.amp_dtype is not None

        # Initialize model with loaded configuration
        self.model = TaskCompletionViT(
            pretrained_model=config.get('model_name', 'vit_b_16'),
            dropout_rate=config.get('dropout', 0.4),
            num_frames=self.num_frames,
            aggregator=config.get('aggregator', 'lstm')
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Create transforms
        _, self.transform = create_transforms(self.image_size, self.num_frames)

        print(f"ProgressEstimator initialized on {self.device} "
              f"with N={self.num_frames} frames from loaded model "
              f"and {self.precision} precision.")

    def predict(self, image: Image.Image) -> float:
        """
        Predicts the task completion for a single image, using cached embeddings
        for temporal context.
        """
        if self.num_frames == 1:
            return self._predict_single(image)
        else:
            return self._predict_sequence(image)

    def _predict_single(self, image: Image.Image) -> float:
        """Prediction for a single frame (N=1)."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with autocast(dtype=self.amp_dtype, enabled=self.autocast_enabled):
                prediction = self.model(img_tensor)
        return prediction.item()

    def _predict_sequence(self, image: Image.Image) -> float:
        """Prediction for a sequence of frames (N>1) with caching."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with autocast(dtype=self.amp_dtype, enabled=self.autocast_enabled):
                embedding = self.model.get_embedding(img_tensor)

        self.embedding_cache.append(embedding.squeeze(0))

        # Pad the cache if it's not full by repeating the oldest frame
        cached_embeddings = list(self.embedding_cache)
        if len(cached_embeddings) < self.num_frames:
            padded_embeddings = ([cached_embeddings[0]] *
                                 (self.num_frames - len(cached_embeddings)))
            padded_embeddings.extend(cached_embeddings)
            cached_embeddings = padded_embeddings

        seq_embeddings = torch.stack(cached_embeddings).unsqueeze(0)

        with torch.no_grad():
            with autocast(dtype=self.amp_dtype, enabled=self.autocast_enabled):
                prediction = self.model.forward_sequence(seq_embeddings)

        return prediction.item()

    def reset(self):
        """Resets the internal embedding cache."""
        self.embedding_cache.clear()
        print("Embedding cache cleared.")


if __name__ == "__main__":
    main()
