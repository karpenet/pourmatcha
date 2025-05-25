"""
Training module for SERL-LeRobot integration.

This module contains:
- Training configuration classes
- Online RL training orchestration
- Reward function implementations
- SERL integration utilities
"""

from .config import TrainingConfig
from .reward_functions import (
    dummy_reward_function,
    create_custom_reward_function
)

__all__ = [
    "TrainingConfig",
    "dummy_reward_function",
    "create_custom_reward_function",
]
