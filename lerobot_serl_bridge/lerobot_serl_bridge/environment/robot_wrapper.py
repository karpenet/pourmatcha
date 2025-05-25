"""
SERL-compatible environment wrapper for LeRobot robots.
"""

import numpy as np
import torch
from typing import Dict, Callable, Any, Tuple, Optional


class SERLRobotEnvironment:
    """SERL-compatible wrapper for LeRobot robots"""

    def __init__(
        self,
        robot,
        reward_fn: Callable[[np.ndarray, np.ndarray], float],
        max_episode_length: int = 200
    ):
        """
        Initialize the SERL robot environment.

        Args:
            robot: LeRobot robot instance
            reward_fn: Function to calculate rewards from observations
            max_episode_length: Maximum steps per episode
        """
        self.robot = robot
        self.reward_fn = reward_fn
        self.max_episode_length = max_episode_length
        self.previous_observation = None
        self.step_count = 0

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observation"""
        if not self.robot.is_connected:
            self.robot.connect()

        # Get initial observation
        obs = self.robot.capture_observation()
        self.previous_observation = obs
        self.step_count = 0
        return self._format_observation(obs)

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Execute action and return (obs, reward, done, info)

        Args:
            action: Action to execute

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Convert action to torch tensor if needed
        if isinstance(action, np.ndarray):
            action_tensor = torch.from_numpy(action).float()
        else:
            action_tensor = torch.tensor(action, dtype=torch.float32)

        # Execute action
        self.robot.send_action(action_tensor)

        # Get new observation
        new_obs = self.robot.capture_observation()

        # Calculate reward using provided function
        prev_image = self._extract_primary_image(self.previous_observation)
        curr_image = self._extract_primary_image(new_obs)
        reward = self.reward_fn(prev_image, curr_image)

        # Check if episode is done
        self.step_count += 1
        done = self.step_count >= self.max_episode_length

        # Create info dict
        info = {
            "step_count": self.step_count,
            "max_episode_length": self.max_episode_length,
        }

        self.previous_observation = new_obs
        return self._format_observation(new_obs), reward, done, info

    def _extract_primary_image(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract the primary image from observation"""
        if obs is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)

        # Try different image keys in order of preference
        image_keys = [
            "observation.images.main",
            "observation.images.webcam",
            "observation.images.laptop",
            "observation.images.phone",
            "observation.image",
            "image"
        ]

        for key in image_keys:
            if key in obs:
                image = obs[key]
                # Convert torch tensor to numpy if needed
                if hasattr(image, 'numpy'):
                    image = image.numpy()
                elif hasattr(image, 'cpu'):
                    image = image.cpu().numpy()

                # Ensure proper shape and dtype
                if isinstance(image, np.ndarray):
                    if image.dtype != np.uint8:
                        # Normalize to 0-255 range if needed
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)
                    return image

        # Default fallback
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def _format_observation(
        self,
        raw_obs: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Format raw robot observation for SERL compatibility.

        Args:
            raw_obs: Raw observation from robot

        Returns:
            Formatted observation dict
        """
        formatted_obs = {}

        # Extract state information
        if "observation.state" in raw_obs:
            state = raw_obs["observation.state"]
            if hasattr(state, 'numpy'):
                state = state.numpy()
            elif hasattr(state, 'cpu'):
                state = state.cpu().numpy()
            formatted_obs["state"] = np.array(state, dtype=np.float32)
        elif "state" in raw_obs:
            state = raw_obs["state"]
            if hasattr(state, 'numpy'):
                state = state.numpy()
            elif hasattr(state, 'cpu'):
                state = state.cpu().numpy()
            formatted_obs["state"] = np.array(state, dtype=np.float32)
        else:
            # Default state if not available (6-DOF for so100)
            formatted_obs["state"] = np.zeros(6, dtype=np.float32)

        # Extract primary image
        formatted_obs["image"] = self._extract_primary_image(raw_obs)

        # Extract all available images with proper keys
        for key, value in raw_obs.items():
            if key.startswith("observation.images."):
                # Extract camera name from observation key
                camera_name = key.split(".")[-1]
                image = value

                # Convert torch tensor to numpy if needed
                if hasattr(image, 'numpy'):
                    image = image.numpy()
                elif hasattr(image, 'cpu'):
                    image = image.cpu().numpy()

                # Ensure proper format
                if isinstance(image, np.ndarray):
                    if image.dtype != np.uint8:
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)

                    # Handle different image formats (C,H,W) vs (H,W,C)
                    if len(image.shape) == 3:
                        if image.shape[0] == 3:  # (C, H, W) format
                            image = np.transpose(
                                image, (1, 2, 0))  # -> (H, W, C)

                    formatted_obs[camera_name] = image

        return formatted_obs

    def close(self):
        """Close the environment and disconnect robot"""
        if self.robot and self.robot.is_connected:
            self.robot.disconnect()

    @property
    def observation_space(self) -> Dict[str, np.ndarray]:
        """Get observation space specification"""
        # Get the actual camera features from the robot if available
        if hasattr(self.robot, 'camera_features'):
            cam_features = self.robot.camera_features
            obs_space = {
                "state": np.zeros(6),  # 6-DOF for so100
                "image": np.zeros((480, 640, 3))  # Default image size
            }

            # Add camera-specific spaces
            for key, feature in cam_features.items():
                if key.startswith("observation.images."):
                    camera_name = key.split(".")[-1]
                    shape = feature.get("shape", (480, 640, 3))
                    obs_space[camera_name] = np.zeros(
                        shape, dtype=np.uint8)

            return obs_space
        else:
            # Fallback for mock robots
            return {
                "state": np.zeros(6),
                "image": np.zeros((480, 640, 3)),
                "main": np.zeros((480, 640, 3)),
                "webcam": np.zeros((480, 640, 3))
            }

    @property
    def action_space(self) -> np.ndarray:
        """Get action space specification"""
        # 6-DOF action space for so100 robot control
        return np.zeros(6)


def create_serl_robot_env(
    robot,
    reward_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    max_episode_length: int = 200
) -> SERLRobotEnvironment:
    """
    Factory function to create SERL-compatible robot environment.

    Args:
        robot: LeRobot robot instance
        reward_fn: Reward function (defaults to dummy function)
        max_episode_length: Maximum steps per episode

    Returns:
        SERLRobotEnvironment instance
    """
    if reward_fn is None:
        from ..training.reward_functions import dummy_reward_function
        reward_fn = dummy_reward_function

    return SERLRobotEnvironment(
        robot=robot,
        reward_fn=reward_fn,
        max_episode_length=max_episode_length
    )
