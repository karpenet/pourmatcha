"""
SERL-LeRobot Integration Library

A Python library for integrating SERL online reinforcement learning 
with LeRobot ACT policies.

This library provides:
- Conversion between PyTorch and JAX policy formats
- Online RL training using SERL algorithms
- Robot environment wrappers
- Reward function interfaces
- Training utilities and monitoring
"""

__version__ = "0.1.0"
__author__ = "SERL-LeRobot Integration Team"

# Core components
from .bridge import PolicyConverter, JAXACTPolicy
from .training import TrainingConfig
from .environment import SERLRobotEnvironment
from .utils import dummy_reward_function

# Make key classes available at package level
__all__ = [
    "PolicyConverter",
    "JAXACTPolicy",
    "TrainingConfig",
    "SERLRobotEnvironment",
    "dummy_reward_function",
]
