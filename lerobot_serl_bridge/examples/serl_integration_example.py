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


def create_so100_robot(mock: bool = False):
    """
    Create a SO100 robot instance.

    Args:
        mock: If True, creates a mock robot for testing.
              If False, creates a real robot connection.

    Returns:
        SO100 robot instance
    """
    # Create robot configuration
    config = So100RobotConfig(mock=mock)

    # Create robot from config
    robot = make_robot_from_config(config)

    return robot


def create_mock_robot():
    """
    Create a mock robot for demonstration purposes.
    For testing without hardware.
    """
    class MockRobot:
        def __init__(self):
            self.is_connected = False

        def connect(self):
            """Mock connect"""
            self.is_connected = True
            logger.info("Mock robot connected")

        def capture_observation(self):
            """Mock observation from robot"""
            import numpy as np
            return {
                "observation.state": np.random.randn(6).astype(np.float32),
                "observation.images.main": np.random.randint(
                    0, 255, (480, 640, 3), dtype=np.uint8),
                "observation.images.webcam": np.random.randint(
                    0, 255, (480, 640, 3), dtype=np.uint8)
            }

        def send_action(self, action):
            """Mock action execution"""
            # In practice, this would send commands to the robot
            import torch
            if isinstance(action, torch.Tensor):
                return action
            else:
                return torch.tensor(action, dtype=torch.float32)

        def disconnect(self):
            """Mock disconnect"""
            self.is_connected = False
            logger.info("Mock robot disconnected")

    return MockRobot()


def main():
    """Main training loop"""
    logger.info("Starting SERL-LeRobot integration example...")

    # Create training configuration
    config = TrainingConfig(
        max_steps=1000,
        max_episode_length=100,
        action_dim=6,  # 6-DOF for so100
        image_keys=["main", "webcam"],
        checkpoint_frequency=500,
        policy_path="/home/jasonx/Dropbox/repos/pourmatcha/pretrained_model"
    )

    # Create robot - set use_mock=True for testing without hardware
    use_mock = True  # Set to False when using real hardware

    if use_mock:
        logger.info("Creating mock robot for testing...")
        robot = create_mock_robot()
    else:
        logger.info("Creating SO100 robot...")
        robot = create_so100_robot(mock=False)

    # Create trainer
    trainer = OnlineRLTrainer(
        config=config,
        robot=robot
    )

    try:
        # Run training
        logger.info("Starting online RL training...")
        trainer.run_training(
            num_iterations=10,  # Number of training iterations
            episodes_per_iteration=5  # Episodes per iteration
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
