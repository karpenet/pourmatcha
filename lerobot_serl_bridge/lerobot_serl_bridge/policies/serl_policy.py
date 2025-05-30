"""
SERL Policy wrapper for LeRobot compatibility.

This module provides a wrapper around SERL agents to make them compatible with 
LeRobot's PreTrainedPolicy interface while maintaining JAX-based computation.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from collections import deque
import torch
from torch import Tensor
import jax
import jax.numpy as jnp

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.configs.types import NormalizationMode
from serl_launcher.agents.continuous.drq import DrQAgent

from .serl_config import SERLConfig

logger = logging.getLogger(__name__)


class SERLPolicy(PreTrainedPolicy):
    """
    SERL Policy wrapper that makes SERL agents compatible with LeRobot.

    This wrapper:
    - Inherits from PreTrainedPolicy for LeRobot compatibility
    - Handles input/output normalization using LeRobot's system
    - Converts between PyTorch tensors (LeRobot) and JAX arrays (SERL)
    - Maintains SERL's JAX-based computation behind the interface
    - Supports saving/loading via LeRobot's hub system
    """

    config_class = SERLConfig
    name = "serl"

    def __init__(
        self,
        config: SERLConfig,
        dataset_stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
        serl_agent: Optional[DrQAgent] = None,
    ):
        """
        Initialize SERL policy wrapper.

        Args:
            config: SERL policy configuration
            dataset_stats: Dataset statistics for normalization
            serl_agent: Pre-initialized SERL agent (optional)
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Setup normalization (LeRobot style)
        self.normalize_inputs = Normalize(
            config.input_features,
            config.normalization_mapping,
            dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features,
            config.normalization_mapping,
            dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features,
            config.normalization_mapping,
            dataset_stats
        )

        # JAX device management
        self.jax_device = jax.devices()[0]
        self.torch_device = torch.device(config.device)

        # SERL agent
        self.serl_agent = serl_agent
        self._agent_initialized = serl_agent is not None

        # Training state
        self.training_rng = jax.random.PRNGKey(42)

        self.reset()

    def _create_dummy_observations(self) -> Dict[str, np.ndarray]:
        """Create dummy observations for agent initialization."""
        dummy_obs = {}

        for key, feature in self.config.input_features.items():
            if key == "observation.state" and self.config.use_proprio:
                dummy_obs["state"] = np.zeros(feature.shape, dtype=np.float32)
            elif key.startswith("observation.images."):
                # Extract the image key
                image_key = key.split(".")[-1]
                if image_key in self.config.image_keys:
                    # Use a reasonable default image size for initialization
                    # SERL works better with specific image formats
                    dummy_obs[image_key] = np.zeros(
                        (84, 84, 3), dtype=np.uint8)  # Standard format

        return dummy_obs

    def _ensure_agent_initialized(self):
        """Initialize SERL agent if not already done."""
        if self._agent_initialized:
            return

        logger.info("Initializing SERL agent...")

        # Create dummy observations for agent initialization
        # Use actual image dimensions from config
        dummy_obs = {}

        # Add state if using proprioception
        if self.config.use_proprio:
            state_dim = None
            for key, feature in self.config.input_features.items():
                if key == "observation.state":
                    state_dim = feature.shape[0]
                    break
            if state_dim:
                dummy_obs["state"] = np.zeros(state_dim, dtype=np.float32)
                logger.info(
                    f"Created dummy state with shape: {dummy_obs['state'].shape}")

        # Add images using actual dimensions from config
        for image_key in self.config.image_keys:
            # Find the corresponding image feature in config
            image_shape = None
            full_key = f"observation.images.{image_key}"

            if full_key in self.config.input_features:
                feature = self.config.input_features[full_key]
                image_shape = feature.shape

                # Convert from LeRobot (C, H, W) to SERL (H, W, C) format
                if len(image_shape) == 3 and image_shape[0] in [1, 3, 4]:
                    # Assume (C, H, W) format
                    h, w, c = image_shape[1], image_shape[2], image_shape[0]
                    dummy_image_shape = (h, w, c)
                else:
                    # Fallback to standard size
                    dummy_image_shape = (224, 224, 3)
            else:
                # Fallback to standard size
                dummy_image_shape = (224, 224, 3)

            dummy_obs[image_key] = np.zeros(dummy_image_shape, dtype=np.uint8)
            logger.info(
                f"Created dummy image '{image_key}' with shape: {dummy_obs[image_key].shape}")

        dummy_actions = np.zeros(self.config.action_dim, dtype=np.float32)
        logger.info(f"Created dummy actions with shape: {dummy_actions.shape}")

        # Create SERL agent
        rng = jax.random.PRNGKey(42)

        try:
            logger.info(
                f"Creating SERL agent with observations: {[(k, v.shape) for k, v in dummy_obs.items()]}")
            self.serl_agent = DrQAgent.create_drq(
                rng=rng,
                observations=dummy_obs,
                actions=dummy_actions,
                encoder_type=self.config.encoder_type,
                shared_encoder=self.config.shared_encoder,
                use_proprio=self.config.use_proprio,
                image_keys=self.config.image_keys,
                critic_network_kwargs=self.config.critic_network_kwargs,
                policy_network_kwargs=self.config.policy_network_kwargs,
                policy_kwargs=self.config.policy_kwargs,
                critic_ensemble_size=self.config.critic_ensemble_size,
                discount=self.config.discount,
                soft_target_update_rate=self.config.soft_target_update_rate,
                target_entropy=self.config.target_entropy,
            )
        except Exception as e:
            logger.error(f"Failed to create SERL agent: {e}")
            logger.error(f"Dummy observations: {dummy_obs}")
            logger.error(f"Config image keys: {self.config.image_keys}")
            logger.error(f"Config use_proprio: {self.config.use_proprio}")
            logger.error(
                f"Config input features: {list(self.config.input_features.keys())}")
            raise

        # Place agent on JAX device
        self.serl_agent = jax.device_put(
            jax.tree_util.tree_map(jnp.array, self.serl_agent),
            device=self.jax_device
        )

        self._agent_initialized = True
        logger.info("SERL agent initialized successfully")

    def get_optim_params(self) -> Dict:
        """
        Return optimization parameters.
        For SERL, this returns the agent's parameters in PyTorch format.
        """
        if not self._agent_initialized:
            return {}

        # Convert JAX agent parameters to PyTorch for compatibility
        # This is a simplified approach - in practice, you might want
        # to handle this differently based on your training setup
        return {}  # SERL handles optimization internally

    def reset(self):
        """Reset policy state."""
        # Initialize any internal state here if needed
        pass

    @torch.no_grad
    def select_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Select action using SERL agent.

        Args:
            batch: Dictionary containing observations

        Returns:
            Single action tensor
        """
        self.eval()
        self._ensure_agent_initialized()

        # Normalize inputs
        batch = self.normalize_inputs(batch)

        # Convert to SERL format
        serl_obs = self._torch_to_serl_obs(batch)

        # Get action from SERL agent
        self.training_rng, action_key = jax.random.split(self.training_rng)
        action = self.serl_agent.sample_actions(
            observations=serl_obs,
            seed=action_key,
            argmax=False  # Use stochastic actions during rollout
        )

        # Convert back to PyTorch
        action = torch.from_numpy(np.asarray(jax.device_get(action)))
        action = action.to(self.torch_device)

        # Unnormalize action
        action_dict = {"action": action.unsqueeze(0)}  # Add batch dim
        action_dict = self.unnormalize_outputs(action_dict)

        return action_dict["action"].squeeze(0)  # Remove batch dim

    def forward(self, batch: Dict[str, Tensor]) -> tuple[Tensor, Dict]:
        """
        Forward pass for training.

        Args:
            batch: Training batch with observations and actions

        Returns:
            Tuple of (loss, info_dict)
        """
        self._ensure_agent_initialized()

        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Convert to SERL format
        serl_batch = self._torch_to_serl_batch(batch)

        # Perform SERL update
        self.serl_agent, update_info = self.serl_agent.update_high_utd(
            serl_batch, utd_ratio=self.config.utd_ratio
        )

        # Extract loss and metrics
        loss = update_info.get("actor_loss", 0.0) + \
            update_info.get("critic_loss", 0.0)
        loss = torch.tensor(float(loss), device=self.torch_device)

        info_dict = {
            "actor_loss": float(update_info.get("actor_loss", 0.0)),
            "critic_loss": float(update_info.get("critic_loss", 0.0)),
            "q_value": float(update_info.get("predicted_qs", 0.0)),
        }

        return loss, info_dict

    def _torch_to_serl_obs(self, batch: Dict[str, Tensor]) -> Dict[str, jnp.ndarray]:
        """Convert PyTorch observation batch to SERL format."""
        serl_obs = {}

        # Track which image keys we've seen vs which ones we need
        available_images = {}

        for key, tensor in batch.items():
            if key == "observation.state" and self.config.use_proprio:
                serl_obs["state"] = jax.device_put(
                    jnp.array(tensor.cpu().numpy()), device=self.jax_device
                )
            elif key.startswith("observation.images."):
                # Extract image key and store for later processing
                image_key = key.split(".")[-1]

                # Convert from LeRobot (B, C, H, W) to SERL (B, H, W, C)
                img_tensor = tensor.cpu().numpy()
                if len(img_tensor.shape) == 4:  # (B, C, H, W)
                    img_tensor = np.transpose(img_tensor, (0, 2, 3, 1))
                elif len(img_tensor.shape) == 3:  # (C, H, W)
                    img_tensor = np.transpose(img_tensor, (1, 2, 0))

                # Get expected shape from config
                expected_shape = None
                if key in self.config.input_features:
                    feature = self.config.input_features[key]
                    if len(feature.shape) == 3:
                        # Convert (C, H, W) to (H, W, C)
                        expected_shape = (
                            feature.shape[1], feature.shape[2], feature.shape[0])

                # Resize image if needed
                if expected_shape and img_tensor.shape[-3:-1] != expected_shape[:2]:
                    logger.info(
                        f"Resizing image from {img_tensor.shape} to match expected {expected_shape}")
                    # Simple resize - in practice you might want to use cv2 or PIL
                    from scipy.ndimage import zoom
                    if len(img_tensor.shape) == 4:  # Batch
                        scale_h = expected_shape[0] / img_tensor.shape[1]
                        scale_w = expected_shape[1] / img_tensor.shape[2]
                        img_tensor = zoom(
                            img_tensor, (1, scale_h, scale_w, 1), order=1)
                    else:  # Single image
                        scale_h = expected_shape[0] / img_tensor.shape[0]
                        scale_w = expected_shape[1] / img_tensor.shape[1]
                        img_tensor = zoom(
                            img_tensor, (scale_h, scale_w, 1), order=1)

                # Convert to uint8 if needed
                if img_tensor.dtype != np.uint8:
                    # Assume input is in [0, 1] range, convert to [0, 255]
                    if img_tensor.max() <= 1.0:
                        img_tensor = (img_tensor * 255).astype(np.uint8)
                    else:
                        img_tensor = img_tensor.astype(np.uint8)

                available_images[image_key] = img_tensor

        # Now ensure all required image keys are present in serl_obs
        for required_key in self.config.image_keys:
            if required_key in available_images:
                # Use the actual image data
                serl_obs[required_key] = jax.device_put(
                    jnp.array(available_images[required_key]), device=self.jax_device
                )
            else:
                # Fallback: use the first available image for missing keys
                # This handles cases where we have "main" but need "webcam"
                if available_images:
                    first_image = list(available_images.values())[0]
                    logger.warning(
                        f"Image key '{required_key}' not found in batch, using first available image")
                    serl_obs[required_key] = jax.device_put(
                        jnp.array(first_image), device=self.jax_device
                    )
                else:
                    logger.error(
                        f"No images available for required key '{required_key}'")

        return serl_obs

    def _torch_to_serl_batch(self, batch: Dict[str, Tensor]) -> Dict[str, jnp.ndarray]:
        """Convert PyTorch training batch to SERL format."""
        serl_batch = {}

        # Handle observations
        observations = {}
        next_observations = {}

        # Track available images for both current and next observations
        available_obs_images = {}
        available_next_obs_images = {}

        for key, tensor in batch.items():
            if key.startswith("observation."):
                tensor_np = tensor.cpu().numpy()

                if key == "observation.state" and self.config.use_proprio:
                    observations["state"] = jax.device_put(
                        jnp.array(tensor_np), device=self.jax_device
                    )
                    # For next observations, we'd need them in the batch
                    # This is a simplified version
                    next_observations["state"] = observations["state"]

                elif key.startswith("observation.images."):
                    image_key = key.split(".")[-1]

                    # Convert from LeRobot (B, C, H, W) to SERL (B, H, W, C)
                    if len(tensor_np.shape) == 4:  # (B, C, H, W)
                        tensor_np = np.transpose(tensor_np, (0, 2, 3, 1))
                    elif len(tensor_np.shape) == 3:  # (C, H, W)
                        tensor_np = np.transpose(tensor_np, (1, 2, 0))

                    # Convert to uint8 if needed
                    if tensor_np.dtype != np.uint8:
                        if tensor_np.max() <= 1.0:
                            tensor_np = (tensor_np * 255).astype(np.uint8)
                        else:
                            tensor_np = tensor_np.astype(np.uint8)

                    available_obs_images[image_key] = tensor_np
                    # For simplicity, use same images for next_observations
                    available_next_obs_images[image_key] = tensor_np

        # Ensure all required image keys are present in observations
        for required_key in self.config.image_keys:
            if required_key in available_obs_images:
                observations[required_key] = jax.device_put(
                    jnp.array(available_obs_images[required_key]), device=self.jax_device
                )
                next_observations[required_key] = jax.device_put(
                    jnp.array(available_next_obs_images[required_key]), device=self.jax_device
                )
            else:
                # Fallback: use the first available image for missing keys
                if available_obs_images:
                    first_image = list(available_obs_images.values())[0]
                    logger.warning(
                        f"Image key '{required_key}' not found in batch, using first available image")
                    observations[required_key] = jax.device_put(
                        jnp.array(first_image), device=self.jax_device
                    )
                    next_observations[required_key] = observations[required_key]
                else:
                    logger.error(
                        f"No images available for required key '{required_key}'")

        # Handle actions, rewards, etc.
        for key, tensor in batch.items():
            if key == "action":
                serl_batch["actions"] = jax.device_put(
                    jnp.array(tensor.cpu().numpy()), device=self.jax_device
                )
            elif key == "reward":
                serl_batch["rewards"] = jax.device_put(
                    jnp.array(tensor.cpu().numpy()), device=self.jax_device
                )
            elif key == "done":
                done_np = tensor.cpu().numpy()
                serl_batch["dones"] = jax.device_put(
                    jnp.array(done_np), device=self.jax_device
                )
                # masks are opposite of dones
                serl_batch["masks"] = jax.device_put(
                    jnp.array(1.0 - done_np), device=self.jax_device
                )

        serl_batch["observations"] = observations
        serl_batch["next_observations"] = next_observations

        return serl_batch

    def save_pretrained(
        self,
        save_directory: str,
        **kwargs
    ):
        """Save SERL agent and LeRobot components."""
        super().save_pretrained(save_directory, **kwargs)

        if self._agent_initialized:
            # Save SERL agent using JAX checkpointing
            from flax.training import checkpoints
            import os

            agent_path = os.path.join(save_directory, "serl_agent")
            checkpoints.save_checkpoint(
                agent_path,
                target=self.serl_agent.state,
                step=0,
                overwrite=True
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        **kwargs
    ):
        """Load SERL policy from pretrained."""
        # Load the base model
        policy = super().from_pretrained(pretrained_name_or_path, **kwargs)

        # Load SERL agent if available
        try:
            from flax.training import checkpoints
            import os

            agent_path = os.path.join(pretrained_name_or_path, "serl_agent")
            if os.path.exists(agent_path):
                # This would need more sophisticated loading logic
                logger.info(f"Loading SERL agent from {agent_path}")
                # TODO: Implement SERL agent loading
        except Exception as e:
            logger.warning(f"Could not load SERL agent: {e}")

        return policy

    @classmethod
    def from_act_policy(
        cls,
        act_policy,
        encoder_type: str = "resnet-pretrained",
        shared_encoder: bool = True,
        critic_ensemble_size: int = 2,
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        utd_ratio: int = 1,
        **kwargs
    ) -> "SERLPolicy":
        """
        Create SERL policy from an existing ACT policy.

        This extracts the configuration, dataset statistics, and other
        environment-specific settings from the ACT policy to create a
        compatible SERL policy wrapper.

        Args:
            act_policy: The loaded ACT policy to base this on
            encoder_type: SERL encoder type to use
            shared_encoder: Whether to share encoder between actor/critic
            critic_ensemble_size: Number of critics in ensemble
            discount: RL discount factor
            soft_target_update_rate: Soft target update rate
            utd_ratio: Update-to-data ratio
            **kwargs: Additional SERL-specific parameters

        Returns:
            SERLPolicy instance with ACT settings preserved
        """
        from .serl_config import SERLConfig

        # Create SERL config from ACT config
        serl_config = SERLConfig.from_act_config(
            act_policy.config,
            encoder_type=encoder_type,
            shared_encoder=shared_encoder,
            critic_ensemble_size=critic_ensemble_size,
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,
            utd_ratio=utd_ratio,
            **kwargs
        )

        # Extract dataset stats from ACT policy's normalization modules
        dataset_stats = cls._extract_dataset_stats_from_policy(act_policy)

        # Create SERL policy with extracted config and stats
        return cls(config=serl_config, dataset_stats=dataset_stats)

    @staticmethod
    def _extract_dataset_stats_from_policy(policy) -> Dict[str, Dict[str, Tensor]]:
        """
        Extract dataset statistics from a LeRobot policy's normalization modules.

        Args:
            policy: LeRobot policy with normalize_inputs and unnormalize_outputs

        Returns:
            Dictionary of dataset statistics for normalization
        """
        import torch

        dataset_stats = {}

        # Extract stats from input normalization
        if hasattr(policy, 'normalize_inputs'):
            for key, feature in policy.normalize_inputs.features.items():
                norm_mode = policy.normalize_inputs.norm_map.get(
                    feature.type, NormalizationMode.IDENTITY)

                if norm_mode == NormalizationMode.IDENTITY:
                    continue

                buffer_name = "buffer_" + key.replace(".", "_")
                if hasattr(policy.normalize_inputs, buffer_name):
                    buffer = getattr(policy.normalize_inputs, buffer_name)
                    dataset_stats[key] = {}

                    for stat_name, param in buffer.items():
                        dataset_stats[key][stat_name] = param.data.clone()

        # Extract stats from output unnormalization
        if hasattr(policy, 'unnormalize_outputs'):
            for key, feature in policy.unnormalize_outputs.features.items():
                norm_mode = policy.unnormalize_outputs.norm_map.get(
                    feature.type, NormalizationMode.IDENTITY)

                if norm_mode == NormalizationMode.IDENTITY:
                    continue

                buffer_name = "buffer_" + key.replace(".", "_")
                if hasattr(policy.unnormalize_outputs, buffer_name):
                    buffer = getattr(policy.unnormalize_outputs, buffer_name)
                    if key not in dataset_stats:
                        dataset_stats[key] = {}

                    for stat_name, param in buffer.items():
                        dataset_stats[key][stat_name] = param.data.clone()

        return dataset_stats
