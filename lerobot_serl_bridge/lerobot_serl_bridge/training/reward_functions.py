"""
Reward functions for SERL-LeRobot training.

This module provides various reward function implementations
for different robot tasks.
"""

import numpy as np
from typing import Callable, Dict, Any


def dummy_reward_function(
    before_image: np.ndarray,
    after_image: np.ndarray
) -> float:
    """
    Dummy reward function as requested.
    Replace this with your actual reward computation logic.

    Args:
        before_image: Image before action execution
        after_image: Image after action execution

    Returns:
        Reward value (currently always 0.5)
    """
    return 0.5


def create_custom_reward_function(
    reward_type: str,
    **kwargs
) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Factory function for creating custom reward functions.

    Args:
        reward_type: Type of reward function ('dummy', 'movement', 'completion')
        **kwargs: Additional parameters for specific reward functions

    Returns:
        Reward function
    """
    if reward_type == "dummy":
        return dummy_reward_function
    elif reward_type == "movement":
        return _create_movement_reward(**kwargs)
    elif reward_type == "completion":
        return _create_completion_reward(**kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def _create_movement_reward(
    movement_threshold: float = 0.1,
    max_reward: float = 1.0
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Create reward function based on image movement/change."""
    def movement_reward(before_image: np.ndarray, after_image: np.ndarray):
        if before_image.shape != after_image.shape:
            return 0.0

        # Calculate pixel-wise difference
        diff = np.mean(np.abs(after_image - before_image))

        # Normalize and clip
        reward = min(diff / movement_threshold, 1.0) * max_reward
        return float(reward)

    return movement_reward


def _create_completion_reward(
    success_threshold: float = 0.8,
    **detector_kwargs
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Create reward function based on task completion detection."""
    def completion_reward(before_image: np.ndarray, after_image: np.ndarray):
        # Placeholder for task completion detection
        # In practice, this would use computer vision to detect
        # task completion (e.g., object picked up, area cleaned, etc.)

        # For now, return based on image difference
        diff = np.mean(np.abs(after_image - before_image))

        # Higher difference suggests more action/progress
        if diff > success_threshold:
            return 1.0
        else:
            return diff / success_threshold

    return completion_reward


def curriculum_reward_wrapper(
    base_reward_fn: Callable[[np.ndarray, np.ndarray], float],
    episode_num: int,
    curriculum_schedule: Dict[int, float]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Wrapper for implementing curriculum learning with reward scaling.

    Args:
        base_reward_fn: Base reward function to wrap
        episode_num: Current episode number
        curriculum_schedule: Dict mapping episode numbers to difficulty multipliers

    Returns:
        Wrapped reward function with curriculum scaling
    """
    # Find the appropriate difficulty multiplier
    multiplier = 1.0
    for threshold, mult in sorted(curriculum_schedule.items()):
        if episode_num >= threshold:
            multiplier = mult
        else:
            break

    def curriculum_reward(before_image: np.ndarray, after_image: np.ndarray):
        base_reward = base_reward_fn(before_image, after_image)
        return base_reward * multiplier

    return curriculum_reward
