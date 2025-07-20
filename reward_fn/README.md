# Vision Transformer for Task Completion Estimation

This experiment trains a Vision Transformer (ViT) to estimate task completion percentage (0-1) from single robot demonstration frames by learning to regress frame indices and normalizing them.

## 🎯 Experiment Overview

**Approach**:

- Regress frame indices from robot demonstration videos
- Normalize frame indices to [0, 1] to get task completion percentage
- Use off-the-shelf ViT with regularization
- Track training with Weights & Biases

**Key Features**:

- **Elegant Solution**: Direct regression from visual features to completion percentage
- **Regularization**: Dropout, L2 regularization, gradient clipping, data augmentation
- **Comprehensive Monitoring**: WandB integration with visualization
- **Production Ready**: Inference scripts and proper checkpointing

## 📊 Dataset

- **Episodes**: 101 robot demonstration episodes (0-100)
- **Resolution**: 640x480 images  
- **Total Frames**: ~25K frames across all episodes
- **Frame Format**: Sequential JPG images (`frame_XXXXXX.jpg`)
- **Labels**: Frame index normalized by episode length → completion percentage

## 🏗️ Architecture

```
Input Image (224x224) → ViT Backbone → Regression Head → Sigmoid → Completion [0,1]
```

**Model Details**:

- **Backbone**: ViT-B/16 (pretrained on ImageNet)
- **Head**: 768 → 384 → 192 → 1 with GELU activations
- **Regularization**: 15% dropout + L2 regularization (1e-4)
- **Output**: Sigmoid activation for [0, 1] range

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_vit.txt
```

### 2. Test Data Loading

```bash
python test_data_loading.py
```

### 3. Run Training

```bash
./run_training.sh
```

### 4. Run Inference

```bash
# Single image
python inference_vit.py --checkpoint checkpoints/best_model.pth --image data/episode_0000/frames/frame_000100.jpg

# Full episode sequence
python inference_vit.py --checkpoint checkpoints/best_model.pth --episode data/episode_0000 --output results.png
```

## 📈 Training Configuration

**Optimized Hyperparameters**:

- **Batch Size**: 24 (memory efficient)
- **Learning Rate**: 3e-4 with cosine annealing
- **Epochs**: 100
- **Dropout**: 15%
- **Weight Decay**: 1e-4
- **Validation Split**: 20%

**Regularization Strategy**:

1. **Data Augmentation**: Random flips, color jitter, rotation
2. **Dropout**: 15% in regression head
3. **L2 Regularization**: 1e-4 on head weights
4. **Gradient Clipping**: Max norm 1.0
5. **Early Stopping**: Based on validation MAE

## 📊 Evaluation Metrics

- **Primary**: Mean Absolute Error (MAE)
- **Secondary**: Mean Squared Error (MSE)
- **Visualization**: Scatter plots of predictions vs ground truth

## 🔬 Technical Details

### Data Preprocessing

```python
# Frame index normalization
completion_pct = frame_idx / (num_frames - 1) if num_frames > 1 else 0.0
completion_pct = min(1.0, max(0.0, completion_pct))  # Clamp to [0, 1]
```

### Shape Commentary (ML Best Practice)

```python
images = batch['image'].to(device)      # Shape: (B, 3, 224, 224)
features = backbone(images)             # Shape: (B, 768)
hidden = linear1(features)              # Shape: (B, 384)  
hidden = linear2(hidden)                # Shape: (B, 192)
output = sigmoid(linear3(hidden))       # Shape: (B, 1) → (B,)
```

### Efficient Design Choices

- **torchvision ViT**: Faster than transformers library
- **Cosine Annealing**: Better convergence than fixed LR
- **Mixed Precision**: Automatic with modern PyTorch
- **Gradient Accumulation**: Effective batch size scaling

## 📁 File Structure

```
reward_fn/
├── train_completion_vit.py      # Main training script
├── inference_vit.py             # Inference and evaluation
├── test_data_loading.py         # Data validation
├── run_training.sh              # Training launcher
├── requirements_vit.txt         # Dependencies
├── README_VIT_EXPERIMENT.md     # This file
├── data/                        # Robot demonstration data
│   ├── collection_metadata.json
│   └── episode_XXXX/
│       ├── metadata.json
│       └── frames/
│           └── frame_XXXXXX.jpg
└── checkpoints/                 # Model checkpoints
    └── best_model.pth
```

## 🎯 Expected Results

**Performance Targets**:

- **MAE < 0.05**: Excellent task completion estimation
- **Smooth Predictions**: Monotonic increase within episodes
- **Generalization**: Good performance across different episode lengths

## 🔧 Advanced Usage

### Custom Training

```bash
python train_completion_vit.py \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --dropout 0.2 \
    --wandb_project "my-experiment"
```

### Model Analysis

```bash
# Generate data statistics
python test_data_loading.py

# Visualize episode predictions  
python inference_vit.py --episode data/episode_0005 --checkpoint checkpoints/best_model.pth
```

## 🚨 Troubleshooting

**Common Issues**:

1. **CUDA OOM**: Reduce batch size or image size
2. **No improvement**: Check learning rate and augmentation strength
3. **Data loading errors**: Verify episode directory structure

**Performance Tips**:

- Monitor validation curves in WandB
- Use gradient clipping to stabilize training
- Increase regularization if overfitting

## 🎨 Why This Approach Works

1. **Visual Progression**: Robot tasks have clear visual progression markers
2. **Temporal Consistency**: Frame index provides natural supervision signal  
3. **Transfer Learning**: ImageNet pretrained features capture relevant visual patterns
4. **Regularization**: Prevents overfitting to specific episode characteristics

## 📚 Extensions

**Future Improvements**:

- **Multi-frame Input**: Use temporal context (3D CNN or transformer)
- **Uncertainty Quantification**: Bayesian neural networks or ensembles
- **Active Learning**: Query most informative frames for labeling
- **Multi-task Learning**: Joint completion + action prediction

---

**Author**: Claude Sonnet 4  
**Date**: January 2025  
**Experiment**: Task Completion Estimation via Frame Index Regression
