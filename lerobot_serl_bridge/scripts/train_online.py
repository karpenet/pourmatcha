#!/usr/bin/env python3
"""
Main training script for SERL-LeRobot online RL.

This script provides a command-line interface for running online RL training
using the lerobot_serl_bridge library.
"""

from lerobot_serl_bridge.training.reward_functions import (
    create_custom_reward_function
)
from lerobot_serl_bridge import TrainingConfig, OnlineRLTrainer
from lerobot.common.robot_devices.control_configs import RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# LeRobot imports

# Library imports

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="SERL Online RL Training for LeRobot ACT Policies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Robot configuration
    parser.add_argument(
        "--robot_type",
        type=str,
        default="so100",
        help="Robot type (so100, etc.)"
    )

    # Policy configuration
    parser.add_argument(
        "--initial_policy_path",
        type=str,
        required=True,
        help="Path to initial PyTorch ACT policy"
    )

    # Training configuration
    parser.add_argument(
        "--training_iterations",
        type=int,
        default=100,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--episodes_per_iteration",
        type=int,
        default=5,
        help="Episodes to collect per iteration"
    )
    parser.add_argument(
        "--max_episode_length",
        type=int,
        default=200,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size"
    )
    parser.add_argument(
        "--buffer_capacity",
        type=int,
        default=100000,
        help="Replay buffer capacity"
    )

    # Checkpointing and output
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=10,
        help="Save checkpoint every N iterations"
    )

    # Reward function
    parser.add_argument(
        "--reward_function",
        type=str,
        default="dummy",
        choices=["dummy", "movement", "completion"],
        help="Reward function to use"
    )

    # Logging
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="serl_lerobot_training",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity"
    )

    return parser.parse_args()


def setup_robot(robot_type: str):
    """Setup and connect to robot"""
    logger.info(f"Setting up robot: {robot_type}")

    # Create robot configuration
    robot_config = RobotConfig()
    robot_config.robot_type = robot_type

    # Initialize robot
    robot = make_robot_from_config(robot_config)

    # Connect to robot
    if not robot.is_connected:
        robot.connect()
        logger.info("Robot connected successfully")

    return robot


def create_training_config(args) -> TrainingConfig:
    """Create training configuration from arguments"""
    return TrainingConfig(
        max_steps=args.training_iterations *
        args.episodes_per_iteration * args.max_episode_length,
        replay_buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        max_episode_length=args.max_episode_length,
        checkpoint_frequency=args.checkpoint_frequency,
        checkpoint_path=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


def main():
    """Main training function"""
    args = parse_args()

    # Validate arguments
    if not os.path.exists(args.initial_policy_path):
        logger.error(
            f"Initial policy path does not exist: {args.initial_policy_path}")
        return 1

    try:
        # Setup robot
        robot = setup_robot(args.robot_type)

        # Create training configuration
        config = create_training_config(args)

        # Setup reward function
        reward_fn = create_custom_reward_function(args.reward_function)

        # Initialize trainer
        trainer = OnlineRLTrainer(
            config=config,
            robot=robot,
            initial_policy_path=args.initial_policy_path
        )

        # Override reward function if not dummy
        if args.reward_function != "dummy":
            trainer.environment.reward_fn = reward_fn

        # Run training
        logger.info("Starting online RL training...")
        trainer.run_training(
            num_iterations=args.training_iterations,
            episodes_per_iteration=args.episodes_per_iteration
        )

        # Save final checkpoint
        trainer.save_checkpoint(args.training_iterations,
                                os.path.join(args.checkpoint_dir, "final"))

        logger.info("Training completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        try:
            trainer.cleanup()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
