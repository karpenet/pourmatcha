"""
Utility functions for SERL integration.
"""

import numpy as np
from typing import Dict, Any
import gymnasium as gym


def setup_serl_environment(
    robot,
    reward_fn,
    max_episode_length: int = 200,
    image_keys: list = None
):
    """
    Setup a SERL-compatible environment from a LeRobot robot.

    Args:
        robot: LeRobot robot instance
        reward_fn: Reward function
        max_episode_length: Maximum episode length
        image_keys: List of image observation keys

    Returns:
        SERL-compatible environment
    """
    from ..environment.robot_wrapper import SERLRobotEnvironment

    if image_keys is None:
        image_keys = ["observation.image"]

    return SERLRobotEnvironment(
        robot=robot,
        reward_fn=reward_fn,
        max_episode_length=max_episode_length
    )


def convert_gym_space(obs_space: Dict[str, Any]) -> gym.Space:
    """
    Convert observation space to gym format for SERL compatibility.

    Args:
        obs_space: Observation space dictionary

    Returns:
        gym.Space compatible observation space
    """
    spaces = {}

    for key, space in obs_space.items():
        if isinstance(space, np.ndarray):
            if key == "image":
                # Image space
                spaces[key] = gym.spaces.Box(
                    low=0, high=255,
                    shape=space.shape,
                    dtype=np.uint8
                )
            else:
                # State space
                spaces[key] = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=space.shape,
                    dtype=np.float32
                )
        else:
            # Default to Box space
            spaces[key] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(len(space),) if hasattr(space, '__len__') else (1,),
                dtype=np.float32
            )

    return gym.spaces.Dict(spaces)


def format_batch_for_serl(batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Format batch data for SERL training.

    Args:
        batch: Batch dictionary with various data types

    Returns:
        Formatted batch with numpy arrays
    """
    formatted_batch = {}

    for key, value in batch.items():
        if isinstance(value, (list, tuple)):
            formatted_batch[key] = np.array(value)
        elif hasattr(value, 'numpy'):
            # PyTorch tensor
            formatted_batch[key] = value.cpu().numpy()
        elif hasattr(value, '__array__'):
            # JAX array or similar
            formatted_batch[key] = np.asarray(value)
        else:
            # Already numpy or scalar
            formatted_batch[key] = np.array(value)

    return formatted_batch


def create_default_serl_config():
    """
    Create default configuration for SERL agent.

    Returns:
        Dictionary with default SERL agent configuration
    """
    return {
        "critic_network_kwargs": {
            "hidden_dims": [256, 256],
        },
        "policy_network_kwargs": {
            "hidden_dims": [256, 256],
        },
        "policy_kwargs": {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        "critic_ensemble_size": 2,
        "critic_subsample_size": None,
        "discount": 0.95,
        "soft_target_update_rate": 0.005,
        "temperature_init": 1.0,
    }


def log_training_metrics(
    wandb_logger,
    metrics: Dict[str, float],
    step: int,
    prefix: str = ""
):
    """
    Log training metrics to wandb.

    Args:
        wandb_logger: Wandb logger instance
        metrics: Dictionary of metrics to log
        step: Training step
        prefix: Optional prefix for metric names
    """
    if wandb_logger is None:
        return

    formatted_metrics = {}
    for key, value in metrics.items():
        metric_name = f"{prefix}/{key}" if prefix else key
        formatted_metrics[metric_name] = float(value)

    try:
        wandb_logger.log(formatted_metrics, step=step)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to log metrics: {e}")
