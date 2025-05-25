#!/usr/bin/env python3
"""
Basic training example for lerobot_serl_bridge.

This script demonstrates how to set up and run basic online RL training
using the lerobot_serl_bridge library.
"""

import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run basic training example."""
    try:
        # Import required modules
        from lerobot_serl_bridge import OnlineRLTrainer
        from lerobot_serl_bridge.config import TrainingConfig
        from lerobot_serl_bridge.utils.reward_functions import dummy_reward_function

        # Note: You'll need to replace this with your actual robot setup
        # from lerobot import Robot
        # robot = Robot("your_robot_type")

        logger.info("Setting up training configuration...")

        # Create training configuration
        config = TrainingConfig(
            # Episode settings
            max_episode_length=50,  # Short episodes for demo
            num_episodes_per_iteration=5,
            num_training_iterations=10,

            # Training settings
            batch_size=16,
            buffer_capacity=1000,
            learning_rate=3e-4,

            # Checkpointing
            checkpoint_frequency=5,
            checkpoint_dir="./example_checkpoints",

            # Logging
            wandb_project="lerobot-serl-example",
            log_frequency=1
        )

        logger.info("Training configuration created successfully")

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # Example policy path (replace with your actual policy)
        policy_path = "path/to/your/act_policy.safetensors"

        if not os.path.exists(policy_path):
            logger.warning(f"Policy file not found: {policy_path}")
            logger.info(
                "Please provide a valid ACT policy file to run training")
            logger.info("Example usage:")
            logger.info(
                "  python basic_training.py --policy-path /path/to/policy.safetensors")
            return

        # Note: Uncomment and modify this section when you have a real robot
        """
        logger.info("Initializing trainer...")
        
        # Initialize trainer
        trainer = OnlineRLTrainer(
            config=config,
            robot=robot,
            initial_policy_path=policy_path,
            reward_function=dummy_reward_function
        )
        
        logger.info("Starting training...")
        
        # Run training
        trainer.run_training()
        
        logger.info("Training completed successfully!")
        
        # Cleanup
        trainer.cleanup()
        """

        logger.info(
            "Example setup completed. Uncomment the training section to run with a real robot.")

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure lerobot_serl_bridge is properly installed")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
