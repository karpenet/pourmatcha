# LeRobot-SERL Bridge

A Python library for integrating SERL (Sample Efficient Reinforcement Learning) online RL with LeRobot ACT (Action Chunking with Transformers) policies.

## Overview

This library provides a bridge between:

- **LeRobot**: A robotics framework with ACT policies implemented in PyTorch
- **SERL**: An online reinforcement learning framework implemented in JAX/FLAX

The integration allows you to:

1. Load pre-trained ACT policies from LeRobot
2. Convert them to JAX format for SERL compatibility
3. Perform online RL training using SERL algorithms
4. Convert improved policies back to PyTorch format

## Features

- **Policy Conversion**: Bidirectional conversion between PyTorch (LeRobot) and JAX (SERL) formats
- **Online Training**: SERL-compatible online RL training loop
- **Robot Environment Wrapper**: SERL-compatible interface for LeRobot robots
- **Flexible Reward Functions**: Customizable reward functions for different tasks
- **Experiment Tracking**: Integration with Weights & Biases and TensorBoard
- **Checkpointing**: Save and resume training with full state preservation

## Installation

### Prerequisites

1. Install LeRobot following their [installation guide](https://github.com/huggingface/lerobot)
2. Install SERL following their [installation guide](https://github.com/rail-berkeley/serl)

### Install the Bridge

```bash
# Clone this repository
git clone https://github.com/yourusername/lerobot_serl_bridge.git
cd lerobot_serl_bridge

# Install in development mode
pip install -e .

# Or install from PyPI (when available)
pip install lerobot_serl_bridge
```

### GPU Support

For CUDA 11:

```bash
pip install "jax[cuda11_pip]>=0.4.20"
```

For CUDA 12:

```bash
pip install "jax[cuda12_pip]>=0.4.20"
```

## Quick Start

### 1. Basic Usage

```python
from lerobot_serl_bridge import OnlineRLTrainer
from lerobot_serl_bridge.config import TrainingConfig
from lerobot import Robot

# Setup robot (replace with your robot type)
robot = Robot("your_robot_type")

# Create training configuration
config = TrainingConfig(
    max_episode_length=100,
    num_episodes_per_iteration=10,
    num_training_iterations=50,
    batch_size=32,
    buffer_capacity=10000
)

# Initialize trainer
trainer = OnlineRLTrainer(
    config=config,
    robot=robot,
    initial_policy_path="path/to/your/act_policy.safetensors"
)

# Run training
trainer.run_training()
```

### 2. Command Line Interface

```bash
# Train with default settings
lerobot-serl-train --robot-type your_robot_type --initial-policy path/to/policy.safetensors

# Train with custom parameters
lerobot-serl-train \
    --robot-type your_robot_type \
    --initial-policy path/to/policy.safetensors \
    --training-iterations 100 \
    --episodes-per-iteration 20 \
    --max-episode-length 200 \
    --checkpoint-dir ./checkpoints \
    --wandb-project my-robot-training
```

### 3. Custom Reward Functions

```python
from lerobot_serl_bridge.utils.reward_functions import RewardFunction

class MyRewardFunction(RewardFunction):
    def __call__(self, observation, action, next_observation, info):
        # Implement your reward logic
        reward = 0.0
        done = False
        
        # Example: reward based on end-effector position
        if 'end_effector_pos' in next_observation:
            target_pos = [0.5, 0.0, 0.3]  # Example target
            current_pos = next_observation['end_effector_pos']
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            reward = -distance
            done = distance < 0.05  # Success threshold
        
        return reward, done

# Use custom reward function
reward_fn = MyRewardFunction()
trainer = OnlineRLTrainer(config, robot, reward_function=reward_fn)
```

## Architecture

### Core Components

1. **Policy Conversion** (`lerobot_serl_bridge.conversion`)
   - `PolicyConverter`: Converts between PyTorch and JAX formats
   - Handles model architecture mapping and parameter conversion

2. **Training** (`lerobot_serl_bridge.training`)
   - `OnlineRLTrainer`: Main training orchestrator
   - `TrainingConfig`: Configuration management

3. **Environment** (`lerobot_serl_bridge.environment`)
   - `SERLRobotEnvironment`: SERL-compatible robot wrapper
   - Handles observation formatting and action execution

4. **Utilities** (`lerobot_serl_bridge.utils`)
   - Reward functions
   - Helper utilities for data processing

### Data Flow

```
LeRobot ACT Policy (PyTorch) 
    ↓ (conversion)
JAX Policy 
    ↓ (online training)
Improved JAX Policy 
    ↓ (conversion)
Improved LeRobot Policy (PyTorch)
```

## Configuration

### Training Configuration

```python
from lerobot_serl_bridge.config import TrainingConfig

config = TrainingConfig(
    # Episode settings
    max_episode_length=100,
    num_episodes_per_iteration=10,
    num_training_iterations=50,
    
    # Training settings
    batch_size=32,
    buffer_capacity=10000,
    learning_rate=3e-4,
    
    # Checkpointing
    checkpoint_frequency=10,
    checkpoint_dir="./checkpoints",
    
    # Logging
    wandb_project="robot-training",
    log_frequency=1
)
```

### Environment Variables

```bash
# Optional: Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Optional: JAX configuration
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
```

## Examples

See the `examples/` directory for complete examples:

- `basic_training.py`: Simple online RL training
- `custom_reward.py`: Using custom reward functions
- `policy_conversion.py`: Converting between formats
- `evaluation.py`: Evaluating trained policies

## Troubleshooting

### Common Issues

1. **JAX/PyTorch Compatibility**
   - Ensure compatible versions are installed
   - Check CUDA versions match between JAX and PyTorch

2. **Memory Issues**
   - Reduce batch size or buffer capacity
   - Set JAX memory fraction: `export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`

3. **Robot Connection**
   - Verify robot is properly connected and configured
   - Check robot-specific dependencies are installed

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Submit a pull request

### Development Setup

```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black lerobot_serl_bridge/
isort lerobot_serl_bridge/

# Lint code
flake8 lerobot_serl_bridge/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{lerobot_serl_bridge,
  title={LeRobot-SERL Bridge: Integration Library for Online RL with ACT Policies},
  author={SERL-LeRobot Integration Team},
  year={2024},
  url={https://github.com/yourusername/lerobot_serl_bridge}
}
```

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) team for the ACT implementation
- [SERL](https://github.com/rail-berkeley/serl) team for the online RL framework
- JAX and PyTorch communities for the excellent frameworks
