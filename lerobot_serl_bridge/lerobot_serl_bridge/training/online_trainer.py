"""
Online RL trainer for SERL-LeRobot integration.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path

# PyTorch imports for policy handling
import torch


# JAX imports for SERL
import jax
import jax.numpy as jnp
from flax.training import checkpoints

# SERL imports (conditional)
from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
import gym.spaces as spaces

# LeRobot imports
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig

# SERL-LeRobot bridge imports
from ..policies import SERLPolicy, SERLConfig

from enum import Enum

from .config import TrainingConfig
from .reward_functions import dummy_reward_function
from ..environment.robot_wrapper import SERLRobotEnvironment

logger = logging.getLogger(__name__)


class InferencePolicy(Enum):
    ACT = "ACT"
    SERL = "SERL"


class OnlineRLTrainer:
    """Online RL trainer that integrates with LeRobot robot control"""

    def __init__(self, config: TrainingConfig, robot=None):
        """Initialize the online RL trainer"""
        self.config = config
        self.robot = robot

        # Setup logging
        import logging
        self.logger = logging.getLogger(__name__)

        self.torch_device = torch.device(self.config.device)
        self.jax_device = jax.devices(self.config.device)[0]

        # Initialize attributes
        self.policy = None
        self.environment = None
        self.serl_agent = None
        self.replay_buffer = None
        self.training_rng = jax.random.PRNGKey(42)
        self.update_steps = 0

        # Initialize components
        self._setup_policy()
        self._setup_environment()
        self._setup_replay_buffer()
        self._setup_serl_agent()

        # Training state
        self.step_count = 0
        self.episode_count = 0

    def _setup_policy(self):
        """Setup the ACT policy for training"""
        # Try to load existing policy
        print(
            f"Loading policy from {self.config.huggingface_repo_id}")
        self.policy = ACTPolicy.from_pretrained(
            self.config.huggingface_repo_id,
        )
        self.policy.to(self.torch_device)

        self._inference_policy = InferencePolicy.ACT

        # Debug: Print policy configuration
        if hasattr(self.policy, 'config'):
            print(f"ACT Policy config:")
            if hasattr(self.policy.config, 'image_features'):
                print(
                    f"  Image features: {self.policy.config.image_features}")
            if hasattr(self.policy.config, 'input_features'):
                print(
                    f"  Input features: {list(self.policy.config.input_features.keys())}")
            if hasattr(self.policy.config, 'input_shapes'):
                print(f"  Input shapes: {self.policy.config.input_shapes}")
        else:
            print("  No config found on policy")

    def _setup_environment(self):
        """Setup SERL-compatible robot environment"""
        print("Setting up robot environment...")

        self.environment = SERLRobotEnvironment(
            robot=self.robot,
            reward_fn=dummy_reward_function,
            max_episode_length=self.config.max_episode_length
        )

        print("Robot environment setup complete")

    def _setup_replay_buffer(self):
        """Setup the replay buffer for storing experience"""

        # Get sample observation to determine proper spaces
        sample_obs = self.environment.reset()

        # Create proper gym spaces if they don't exist
        obs_spaces = {}
        for key, value in sample_obs.items():
            if key == "state":
                obs_spaces[key] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=value.shape, dtype=np.float32
                )
            elif key == "image":
                obs_spaces[key] = spaces.Box(
                    low=0, high=255, shape=value.shape, dtype=np.uint8
                )

        observation_space = spaces.Dict(obs_spaces)

        # Create action space
        action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.config.action_dim,), dtype=np.float32
        )

        image_keys = ["image"] if "image" in sample_obs else []

        self.replay_buffer = MemoryEfficientReplayBufferDataStore(
            observation_space=observation_space,
            action_space=action_space,
            capacity=self.config.replay_buffer_capacity,
            image_keys=image_keys,
        )

        logger.info(
            f"Replay buffer created with capacity: "
            f"{self.config.replay_buffer_capacity}")

    def _setup_serl_agent(self):
        """Initialize the SERL policy wrapper using ACT policy config"""

        print("Setting up SERL policy wrapper from ACT config...")

        # Create SERL policy wrapper from the loaded ACT policy
        # This preserves all input/output features, normalization settings,
        # and dataset statistics from the ACT policy
        self.serl_policy = SERLPolicy.from_act_policy(
            act_policy=self.policy,
            encoder_type="resnet-pretrained",
            shared_encoder=True,
            critic_ensemble_size=2,
            discount=0.95,
            soft_target_update_rate=0.005,
            utd_ratio=self.config.utd_ratio,
        )

        # Move to appropriate device
        self.serl_policy.to(self.torch_device)

        print("SERL policy wrapper initialized successfully from ACT config")

        # Debug: Print configuration comparison
        print(
            f"ACT config features: {list(self.policy.config.input_features.keys())}")
        print(
            f"SERL config features: {list(self.serl_policy.config.input_features.keys())}")
        print(f"SERL action dim: {self.serl_policy.config.action_dim}")
        print(f"SERL image keys: {self.serl_policy.config.image_keys}")
        print(f"SERL use_proprio: {self.serl_policy.config.use_proprio}")

    def _decide_inference_policy(self):
        self._inference_policy = InferencePolicy.SERL
        # if (len(self.replay_buffer) < self.config.min_buffer_size) or self.update_steps < 1:
        #     self._inference_policy = InferencePolicy.ACT
        # elif jax.random.uniform(self.training_rng) < self.config.policy_switch_probability:
        #     self._inference_policy = InferencePolicy.SERL
        # else:
        #     self._inference_policy = InferencePolicy.ACT

    def collect_experience(
        self, num_episodes: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Collect experience episodes using current policy.

        Args:
            num_episodes: Number of episodes to collect

        Returns:
            List of episode data
        """
        print(f"Collecting {num_episodes} episodes of experience...")

        all_episodes = []

        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")

            # Determine which policy will be used
            self._decide_inference_policy()
            buffer_size = len(self.replay_buffer) if self.replay_buffer else 0
            print(
                f"Using {self._inference_policy.value} policy | Buffer size: {buffer_size} | Update steps: {self.update_steps}")

            # Reset environment
            obs = self.environment.reset()
            episode_data = []
            episode_return = 0.0

            # Collect episode
            for _step in range(self.config.max_episode_length):
                # Get action from policy
                action = self._get_action(obs)

                # Execute action
                next_obs, reward, done, info = self.environment.step(action)

                # Store transition
                transition = {
                    "observation": obs,
                    "action": action,
                    "reward": reward,
                    "next_observation": next_obs,
                    "done": done,
                    "info": info
                }
                episode_data.append(transition)
                episode_return += reward

                # Update observation
                obs = next_obs

                if done:
                    break

            all_episodes.append(episode_data)
            print(
                f"Episode {episode + 1} completed: {len(episode_data)} "
                f"steps, return: {episode_return:.3f}")

            # Add episode to replay buffer if available
            if self.replay_buffer is not None:
                self._add_episode_to_buffer(episode_data)

        return all_episodes

    def _get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Get action from current policy with bootstrap strategy"""

        if (self._inference_policy == InferencePolicy.ACT and
                self.policy is not None):
            # Use ACT policy for bootstrapping
            policy_obs = self._format_observation_for_policy(observation)

            with torch.no_grad():
                # Get action from ACT policy
                action_dict = self.policy.select_action(policy_obs)
                action = action_dict.cpu().numpy()[0]  # Remove batch dim

            logger.debug("Using ACT policy for action selection")
            print(f"ACT Action: {action}")
            return action

        else:
            # Use SERL policy wrapper for action selection
            serl_obs = self._format_observation_for_serl_wrapper(observation)

            with torch.no_grad():
                action_tensor = self.serl_policy.select_action(serl_obs)
                action = action_tensor.cpu().numpy()

            logger.debug("Using SERL policy wrapper for action selection")
            print(f"SERL Action: {action}")
            return action

        # Fallback to zero action if nothing works
        logger.warning("No policy available, using zero action")
        return np.zeros(self.config.action_dim)

    def _format_observation_for_policy(
        self, observation: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Format observation for LeRobot policy"""

        formatted_obs = {}

        if "state" in observation:
            formatted_obs["observation.state"] = torch.FloatTensor(
                observation["state"]).unsqueeze(0).to(self.torch_device)

        if "image" in observation:
            # Ensure image is in correct format (C, H, W)
            image = observation["image"]
            if len(image.shape) == 3 and image.shape[-1] == 3:
                # (H, W, C) -> (C, H, W)
                image = np.transpose(image, (2, 0, 1))

            # Based on your ACT config, provide both main and webcam images
            # Since environment only provides one image, use it for both
            image_tensor = torch.FloatTensor(
                image).unsqueeze(0).to(self.torch_device)
            formatted_obs["observation.images.main"] = image_tensor
            formatted_obs["observation.images.webcam"] = image_tensor

        return formatted_obs

    def _format_observation_for_serl_wrapper(
        self, observation: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Format observation for SERL policy wrapper (PyTorch tensors)"""

        formatted_obs = {}

        if "state" in observation:
            formatted_obs["observation.state"] = torch.FloatTensor(
                observation["state"]).unsqueeze(0).to(self.torch_device)

        if "image" in observation:
            # Ensure image is in correct format for SERL wrapper
            image = observation["image"]

            # Handle different input formats
            if len(image.shape) == 3:
                if image.shape[-1] == 3:  # (H, W, C)
                    # Convert (H, W, C) -> (C, H, W) for LeRobot format
                    image = np.transpose(image, (2, 0, 1))
                # else already in (C, H, W) format

            # Provide ALL required image keys using the same image data
            # This matches what we do for ACT policy formatting
            image_tensor = torch.FloatTensor(image).unsqueeze(0).to(
                self.torch_device)

            for image_key in self.serl_policy.config.image_keys:
                key = f"observation.images.{image_key}"
                if key in self.serl_policy.config.input_features:
                    formatted_obs[key] = image_tensor

        return formatted_obs

    def _add_episode_to_buffer(self, episode_data: List[Dict[str, Any]]):
        """Add episode data to replay buffer"""
        if self.replay_buffer is None:
            return

        try:
            for transition in episode_data:
                # Format transition for SERL buffer
                # Only include keys that both environment and buffer understand
                obs = transition["observation"]
                next_obs = transition["next_observation"]

                # Create clean observations with only the keys we need
                clean_obs = {}
                clean_next_obs = {}

                if "state" in obs:
                    clean_obs["state"] = obs["state"]
                    clean_next_obs["state"] = next_obs["state"]

                if "image" in obs:
                    clean_obs["image"] = obs["image"]
                    clean_next_obs["image"] = next_obs["image"]

                formatted_transition = {
                    "observations": clean_obs,
                    "actions": transition["action"],
                    "rewards": np.array([transition["reward"]]),
                    "next_observations": clean_next_obs,
                    "masks": np.array([not transition["done"]]),
                    "dones": np.array([transition["done"]])
                }

                # Add to buffer
                self.replay_buffer.insert(formatted_transition)

        except Exception as e:
            logger.error(f"Failed to add episode to buffer: {e}")
            # Log more details about the error
            if len(episode_data) > 0:
                sample_obs = episode_data[0]["observation"]
                logger.error(
                    f"Sample observation keys: {list(sample_obs.keys())}")
            raise e  # Re-raise to see the full error

    def train_policy(self) -> Dict[str, float]:
        """
        Train policy using SERL algorithms through the wrapper.

        Returns:
            Training metrics
        """

        if len(self.replay_buffer) < self.config.batch_size:
            logger.warning("Not enough data in replay buffer for training")
            return {}

        print("Training policy with SERL wrapper...")

        try:
            # Sample batch from replay buffer
            batch = self.replay_buffer.sample(self.config.batch_size)

            # Convert SERL replay buffer format to LeRobot format
            torch_batch = self._convert_serl_batch_to_lerobot(batch)

            # Perform SERL update through wrapper
            loss, update_info = self.serl_policy.forward(torch_batch)

            self.update_steps += 1

            # Extract metrics
            metrics = {
                "total_loss": float(loss.item()),
                "actor_loss": update_info.get("actor_loss", 0.0),
                "critic_loss": update_info.get("critic_loss", 0.0),
                "q_value": update_info.get("q_value", 0.0),
                "update_steps": self.update_steps,
            }

            # Log to wandb if available
            if hasattr(self, 'wandb_logger') and self.wandb_logger is not None:
                self.wandb_logger.log(metrics, step=self.update_steps)

            return metrics

        except Exception as e:
            logger.error(f"Training step failed: {e}")
            return {}

    def _convert_serl_batch_to_lerobot(
        self, serl_batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert SERL replay buffer batch to LeRobot format.

        Args:
            serl_batch: Batch from SERL replay buffer

        Returns:
            Batch in LeRobot format for SERL policy wrapper
        """
        torch_batch = {}

        # Handle observations
        observations = serl_batch.get("observations", {})
        for obs_key, obs_value in observations.items():
            if obs_key == "state":
                torch_batch["observation.state"] = torch.FloatTensor(
                    obs_value).to(self.torch_device)
            elif obs_key in self.serl_policy.config.image_keys:
                # Map to the correct LeRobot image key format
                lerobot_key = f"observation.images.{obs_key}"
                torch_batch[lerobot_key] = torch.FloatTensor(
                    obs_value).to(self.torch_device)

        # Handle actions
        if "actions" in serl_batch:
            torch_batch["action"] = torch.FloatTensor(
                serl_batch["actions"]).to(self.torch_device)

        # Handle rewards (optional)
        if "rewards" in serl_batch:
            torch_batch["reward"] = torch.FloatTensor(
                serl_batch["rewards"]).to(self.torch_device)

        return torch_batch

    def save_checkpoint(self, iteration: int, save_path: Optional[str] = None):
        """Save training checkpoint"""
        if save_path is None:
            save_path = os.path.join(
                self.config.checkpoint_path,
                f"checkpoint_{iteration}"
            )

        os.makedirs(save_path, exist_ok=True)

        try:
            policy_path = os.path.join(save_path, "policy.pth")
            torch.save(self.policy.state_dict(), policy_path)

            # Save SERL agent
            jax_policy_path = os.path.join(save_path, "serl_agent")
            checkpoints.save_checkpoint(
                jax_policy_path,
                target=self.serl_agent.state,
                step=iteration,
                overwrite=True
            )

            # Save training metadata
            metadata = {
                "iteration": iteration,
                "config": self.config.__dict__,
                "timestamp": time.time(),
                "update_steps": self.update_steps,
            }

            import json
            with open(os.path.join(save_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Checkpoint saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def run_training(
        self,
        num_iterations: int,
        episodes_per_iteration: int = 5
    ):
        """
        Run the full online RL training loop.

        Args:
            num_iterations: Number of training iterations
            episodes_per_iteration: Episodes to collect per iteration
        """
        print("Starting online RL training...")

        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_path, exist_ok=True)

        for iteration in range(num_iterations):
            print(f"Training iteration {iteration + 1}/{num_iterations}")

            # Collect experience
            self.collect_experience(num_episodes=episodes_per_iteration)

            metrics = self.train_policy()
            print(f"Training metrics: {metrics}")

            # Save checkpoint
            checkpoint_freq = (self.config.checkpoint_frequency
                               // episodes_per_iteration)
            if (iteration + 1) % checkpoint_freq == 0:
                self.save_checkpoint(iteration + 1)

        print("Online training completed!")

    def cleanup(self):
        """Cleanup resources"""
        if self.environment:
            self.environment.close()

        if self.robot and self.robot.is_connected:
            self.robot.disconnect()
            print("Robot disconnected")
