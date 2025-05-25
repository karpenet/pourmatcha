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
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# JAX imports for SERL
try:
    import jax
    import jax.numpy as jnp
    from flax.training import checkpoints
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# SERL imports (conditional)
try:
    from serl_launcher.agents.continuous.sac import SACAgent
    SERL_AVAILABLE = True
except ImportError:
    logging.warning(
        "SERL not available. Install serl_launcher for full functionality.")
    SERL_AVAILABLE = False

# LeRobot imports
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig

from .config import TrainingConfig
from .reward_functions import dummy_reward_function
from ..environment.robot_wrapper import SERLRobotEnvironment

logger = logging.getLogger(__name__)


class OnlineRLTrainer:
    """Online RL trainer that integrates with LeRobot robot control"""

    def __init__(self, config: TrainingConfig, robot=None):
        """Initialize the online RL trainer"""
        self.config = config
        self.robot = robot

        # Setup logging
        import logging
        self.logger = logging.getLogger(__name__)

        # Initialize attributes
        self.policy = None
        self.environment = None
        self.serl_agent = None
        self.replay_buffer = None
        self.training_rng = jax.random.PRNGKey(42) if JAX_AVAILABLE else None
        self.update_steps = 0

        # Initialize components
        self._setup_policy()
        self._setup_environment()
        self._setup_serl_agent()

        # Training state
        self.step_count = 0
        self.episode_count = 0

    def _setup_policy(self):
        """Setup the ACT policy for training"""
        try:
            # Try to load existing policy
            if (self.config.policy_path and
                    Path(self.config.policy_path).exists()):
                self.logger.info(
                    f"Loading policy from {self.config.policy_path}")
                self.policy = ACTPolicy.from_pretrained(
                    self.config.policy_path)
            else:
                self.logger.info("Creating new ACT policy")
                self.policy = self._create_default_act_policy()
        except Exception as e:
            self.logger.error(f"Failed to setup policy: {e}")
            raise

    def _create_default_act_policy(self):
        """Create a default ACT policy with proper feature configuration"""
        from lerobot.configs.types import PolicyFeature, FeatureType

        # Create ACT configuration with proper features
        act_config = ACTConfig()

        # Configure input features based on robot capabilities
        input_features = {}

        # State features - required for robot control
        input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(self.config.action_dim,)
        )

        # Add image features if available
        if self.config.image_keys:
            for image_key in self.config.image_keys:
                # Default image shape for so100 (C, H, W)
                key = f"observation.images.{image_key}"
                input_features[key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, 480, 640)
                )

        # Output features
        output_features = {
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.config.action_dim,)
            )
        }

        # Set features in config
        act_config.input_features = input_features
        act_config.output_features = output_features

        # Configure input/output shapes for backward compatibility
        act_config.input_shapes = {
            key: list(feature.shape)
            for key, feature in input_features.items()
        }
        act_config.output_shapes = {
            key: list(feature.shape)
            for key, feature in output_features.items()
        }

        # Configure normalization modes
        act_config.input_normalization_modes = {}
        act_config.output_normalization_modes = {}

        for key, feature in input_features.items():
            if feature.type == FeatureType.VISUAL:
                act_config.input_normalization_modes[key] = "mean_std"
            else:
                act_config.input_normalization_modes[key] = "mean_std"

        for key, feature in output_features.items():
            act_config.output_normalization_modes[key] = "mean_std"

        # Vision encoder configuration
        if self.config.image_keys:
            act_config.vision_backbone = "resnet18"
            pretrained_weights = "ResNet18_Weights.IMAGENET1K_V1"
            act_config.pretrained_backbone_weights = pretrained_weights

        # Other ACT-specific configurations
        act_config.n_action_steps = 100  # Horizon for action prediction
        act_config.chunk_size = 50  # Size of action chunks
        act_config.n_obs_steps = 1

        # Create the policy
        policy = ACTPolicy(act_config)

        # Initialize with dummy stats to avoid "mean is infinity" error
        try:
            # Create dummy stats for normalization
            dummy_stats = {}
            for key, feature in act_config.input_features.items():
                if feature.type.value == "VISUAL":
                    # Image normalization stats (mean=0, std=1)
                    dummy_stats[key] = {
                        "mean": torch.zeros(feature.shape),
                        "std": torch.ones(feature.shape)
                    }
                else:
                    # State normalization stats (mean=0, std=1)
                    dummy_stats[key] = {
                        "mean": torch.zeros(feature.shape),
                        "std": torch.ones(feature.shape)
                    }

            # Set the stats if the policy supports it
            if hasattr(policy, 'set_stats'):
                policy.set_stats(dummy_stats)
            elif hasattr(policy, 'stats'):
                policy.stats = dummy_stats

        except Exception as e:
            self.logger.warning(f"Could not initialize policy stats: {e}")

        self.logger.info(f"Created ACT policy with config: {act_config}")
        return policy

    def _setup_environment(self):
        """Setup SERL-compatible robot environment"""
        logger.info("Setting up robot environment...")

        self.environment = SERLRobotEnvironment(
            robot=self.robot,
            reward_fn=dummy_reward_function,
            max_episode_length=self.config.max_episode_length
        )

        logger.info("Robot environment setup complete")

    def _setup_serl_agent(self):
        """Initialize the SERL SAC agent"""
        if not (SERL_AVAILABLE and JAX_AVAILABLE):
            return

        try:
            # Create dummy observations and actions for agent initialization
            dummy_obs = {
                "state": np.zeros((1, 7), dtype=np.float32),
                "image": np.zeros((1, 224, 224, 3), dtype=np.float32)
            }
            dummy_actions = np.zeros(
                (1, self.config.action_dim), dtype=np.float32)

            # Create the SERL agent using the create_pixels method
            rng = jax.random.PRNGKey(42)

            # Use a simple encoder for the visual features
            from serl_launcher.networks.encoders import encoders
            encoder_def = encoders["resnet_v1_34"](
                pooling_method="avg", normalize=True
            )

            self.serl_agent = SACAgent.create_pixels(
                rng=rng,
                observations=dummy_obs,
                actions=dummy_actions,
                encoder_def=encoder_def,
                shared_encoder=True,
                use_proprio=True,
                critic_network_kwargs={
                    "hidden_dims": [256, 256],
                },
                policy_network_kwargs={
                    "hidden_dims": [256, 256],
                },
                policy_kwargs={
                    "tanh_squash_distribution": True,
                    "std_parameterization": "uniform",
                },
                critic_ensemble_size=2,
                discount=0.95,
                soft_target_update_rate=0.005,
                target_entropy=-self.config.action_dim,
            )

            print("SERL agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SERL agent: {e}")
            self.serl_agent = None

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
        logger.info(f"Collecting {num_episodes} episodes of experience...")

        all_episodes = []

        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes}")

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
            logger.info(
                f"Episode {episode + 1} completed: {len(episode_data)} "
                f"steps, return: {episode_return:.3f}")

            # Add episode to replay buffer if available
            if self.replay_buffer is not None:
                self._add_episode_to_buffer(episode_data)

        return all_episodes

    def _get_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Get action from current policy"""
        if self.serl_agent is not None:
            # Use SERL agent for action selection
            try:
                # Format observation for SERL
                serl_obs = self._format_observation_for_serl(observation)

                # Sample action from SERL agent
                if self.training_rng is not None:
                    self.training_rng, action_key = jax.random.split(
                        self.training_rng)
                    action = self.serl_agent.sample_actions(
                        observations=serl_obs,
                        seed=action_key,
                        argmax=False
                    )
                    return np.asarray(jax.device_get(action))
            except Exception as e:
                logger.warning(f"Failed to get action from SERL agent: {e}")

        # Fallback to PyTorch policy or random actions
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for policy inference")
            return np.random.uniform(-1, 1, self.config.action_dim)

        try:
            # Format observation for policy
            policy_input = self._format_observation_for_policy(observation)

            # Get action from policy
            with torch.no_grad():
                action = self.policy.select_action(policy_input)

            # Convert to numpy
            if hasattr(action, 'numpy'):
                action = action.numpy()
            elif hasattr(action, 'cpu'):
                action = action.cpu().numpy()

            return action
        except Exception as e:
            logger.error(f"Failed to get action from policy: {e}")
            # Return random action as fallback
            return np.random.uniform(-1, 1, self.config.action_dim)

    def _format_observation_for_policy(
        self, observation: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Format observation for LeRobot policy"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        formatted_obs = {}

        if "state" in observation:
            formatted_obs["observation.state"] = torch.FloatTensor(
                observation["state"]).unsqueeze(0)

        if "image" in observation:
            # Ensure image is in correct format (C, H, W)
            image = observation["image"]
            if len(image.shape) == 3 and image.shape[-1] == 3:
                # (H, W, C) -> (C, H, W)
                image = np.transpose(image, (2, 0, 1))
            formatted_obs["observation.image"] = torch.FloatTensor(
                image).unsqueeze(0)

        return formatted_obs

    def _format_observation_for_serl(
        self, observation: Dict[str, np.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Format observation for SERL agent"""
        if not JAX_AVAILABLE:
            raise ImportError("JAX not available")

        formatted_obs = {}

        if "state" in observation:
            formatted_obs["state"] = jnp.array(observation["state"]).to(
                jax.devices("gpu")[0])

        if "image" in observation:
            # Ensure image is in correct format for SERL
            image = observation["image"]
            if len(image.shape) == 3:
                # Add batch dimension if needed
                image = image[None, ...]
            formatted_obs["image"] = jnp.array(image).to(
                jax.devices("gpu")[0])

        return formatted_obs

    def _add_episode_to_buffer(self, episode_data: List[Dict[str, Any]]):
        """Add episode data to replay buffer"""
        if self.replay_buffer is None:
            return

        try:
            for transition in episode_data:
                # Format transition for SERL buffer
                formatted_transition = {
                    "observations": transition["observation"],
                    "actions": transition["action"],
                    "rewards": np.array([transition["reward"]]),
                    "next_observations": transition["next_observation"],
                    "masks": np.array([not transition["done"]]),
                    "dones": np.array([transition["done"]])
                }

                # Add to buffer
                self.replay_buffer.insert(formatted_transition)

        except Exception as e:
            logger.error(f"Failed to add episode to buffer: {e}")

    def train_policy(self) -> Dict[str, float]:
        """
        Train policy using SERL algorithms.

        Returns:
            Training metrics
        """
        if not (SERL_AVAILABLE and self.serl_agent is not None):
            logger.error("SERL agent not available for training")
            return {}

        if len(self.replay_buffer) < self.config.batch_size:
            logger.warning("Not enough data in replay buffer for training")
            return {}

        logger.info("Training policy with SERL...")

        try:
            # Sample batch from replay buffer
            batch = self.replay_buffer.sample(self.config.batch_size)

            # Convert batch to JAX format if needed
            jax_batch = {}
            for key, value in batch.items():
                if isinstance(value, np.ndarray):
                    jax_batch[key] = jnp.array(value)
                else:
                    jax_batch[key] = value

            # Perform SERL update
            self.serl_agent, update_info = self.serl_agent.update_high_utd(
                jax_batch,
                utd_ratio=self.config.utd_ratio
            )

            self.update_steps += 1

            # Extract metrics
            metrics = {
                "policy_loss": float(update_info.get("actor_loss", 0.0)),
                "critic_loss": float(update_info.get("critic_loss", 0.0)),
                "q_value": float(update_info.get("predicted_qs", 0.0)),
                "update_steps": self.update_steps,
            }

            # Log to wandb if available
            if self.wandb_logger is not None:
                self.wandb_logger.log(metrics, step=self.update_steps)

            return metrics

        except Exception as e:
            logger.error(f"Training step failed: {e}")
            return {}

    def save_checkpoint(self, iteration: int, save_path: Optional[str] = None):
        """Save training checkpoint"""
        if save_path is None:
            save_path = os.path.join(
                self.config.checkpoint_path,
                f"checkpoint_{iteration}"
            )

        os.makedirs(save_path, exist_ok=True)

        try:
            # Save PyTorch policy
            if hasattr(self.policy, 'state_dict') and TORCH_AVAILABLE:
                policy_path = os.path.join(save_path, "policy.pth")
                torch.save(self.policy.state_dict(), policy_path)

            # Save SERL agent
            if self.serl_agent is not None and JAX_AVAILABLE:
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

            logger.info(f"Checkpoint saved to {save_path}")

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
        logger.info("Starting online RL training...")

        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_path, exist_ok=True)

        for iteration in range(num_iterations):
            logger.info(f"Training iteration {iteration + 1}/{num_iterations}")

            # Collect experience
            self.collect_experience(num_episodes=episodes_per_iteration)

            # Train policy (if SERL is available and we have enough data)
            if SERL_AVAILABLE and iteration > 0:
                metrics = self.train_policy()
                logger.info(f"Training metrics: {metrics}")

            # Save checkpoint
            checkpoint_freq = (self.config.checkpoint_frequency //
                               episodes_per_iteration)
            if (iteration + 1) % checkpoint_freq == 0:
                self.save_checkpoint(iteration + 1)

        logger.info("Online training completed!")

    def cleanup(self):
        """Cleanup resources"""
        if self.environment:
            self.environment.close()

        if self.robot and self.robot.is_connected:
            self.robot.disconnect()
            logger.info("Robot disconnected")
