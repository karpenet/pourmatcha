"""
Training configuration for SERL-LeRobot integration.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Configuration for SERL training"""
    # RL Training parameters
    replay_buffer_capacity: int = 1000000
    batch_size: int = 256
    utd_ratio: int = 4  # Updates to data ratio

    # Bootstrap parameters for ACT-to-SERL transition
    min_buffer_size: int = 50000  # Minimum buffer size before switching to SERL
    policy_switch_probability: float = 0.5  # Probability of switching to to baseline policy once SERL is used

    # Environment parameters
    max_episode_length: int = 3000
    num_eval_episodes: int = 5
    eval_frequency: int = 1000

    # Policy parameters
    action_dim: int = 7  # Adjust based on your robot
    observation_keys: Optional[List[str]] = None
    image_keys: Optional[List[str]] = None
    policy_path: Optional[str] = None  # Path to existing policy to load

    # Checkpointing
    checkpoint_frequency: int = 5000
    checkpoint_path: str = "serl_act_training"
    huggingface_repo_id: str = "jchun/act_cleaning_16k"

    # Logging
    wandb_project: str = "serl_act_training"
    wandb_entity: Optional[str] = None

    device: str = "cuda"

    def __post_init__(self):
        if self.observation_keys is None:
            self.observation_keys = ["state"]
        if self.image_keys is None:
            self.image_keys = ["observation.image"]
