#!/usr/bin/env python3
"""
SERL Integration Example

This script demonstrates how to use the lerobot_serl_bridge library
to perform online RL training with SERL on a LeRobot robot.
"""

from lerobot_serl_bridge.training.config import TrainingConfig
from lerobot_serl_bridge.training.online_trainer import OnlineRLTrainer
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
import logging
import sys
import os
from pathlib import Path

# Add the package to the path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add LeRobot to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lerobot"))

# LeRobot imports for so100 robot

# SERL bridge imports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# change cwd to ../lerobot
os.chdir(Path(__file__).parent.parent.parent / "lerobot")


def main():
    """Main training loop"""
    logger.info("Starting SERL-LeRobot integration example...")

    # Create training configuration
    config = TrainingConfig(
        max_episode_length=3000,
        action_dim=6,  # 6-DOF for so100
        image_keys=["main", "webcam"],
        checkpoint_frequency=500,
        huggingface_repo_id="jchun/act_cleaning_16k",
        min_buffer_size=3000 * 16,
        policy_switch_probability=0.0,
        replay_buffer_capacity=1000000,
    )

    logger.info("Creating SO100 robot...")
    robot = make_robot_from_config(So100RobotConfig())

    # Create trainer
    trainer = OnlineRLTrainer(
        config=config,
        robot=robot
    )

    try:
        # Run training
        logger.info("Starting online RL training...")
        trainer.run_training(
            num_iterations=1000,  # Number of training iterations
            episodes_per_iteration=16  # Episodes per iteration
        )

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        trainer.cleanup()


if __name__ == "__main__":
    main()
