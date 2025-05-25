"""
Training configuration for SERL-LeRobot integration.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Configuration for SERL training"""
    # RL Training parameters
    max_steps: int = 10000
    replay_buffer_capacity: int = 100000
    batch_size: int = 256
    utd_ratio: int = 4  # Updates to data ratio

    # Environment parameters
    max_episode_length: int = 200
    num_eval_episodes: int = 10
    eval_frequency: int = 1000

    # Policy parameters
    action_dim: int = 7  # Adjust based on your robot
    observation_keys: Optional[List[str]] = None
    image_keys: Optional[List[str]] = None
    policy_path: Optional[str] = None  # Path to existing policy to load

    # Checkpointing
    checkpoint_frequency: int = 5000
    checkpoint_path: str = "./checkpoints/serl_act"

    # Logging
    wandb_project: str = "serl_act_training"
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        if self.observation_keys is None:
            self.observation_keys = ["state"]
        if self.image_keys is None:
            self.image_keys = ["observation.image"]
