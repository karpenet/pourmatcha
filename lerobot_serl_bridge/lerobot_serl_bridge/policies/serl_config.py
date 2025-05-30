"""
Configuration for SERL Policy wrapper.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode, FeatureType, PolicyFeature
from lerobot.common.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("serl")
@dataclass
class SERLConfig(PreTrainedConfig):
    """Configuration class for SERL Policy wrapper.

    This configuration allows SERL agents to be compatible with LeRobot's 
    policy interface while maintaining JAX-based computation.

    Args:
        action_dim: Dimension of the action space
        encoder_type: Type of encoder for visual observations 
        shared_encoder: Whether to share encoder between critic and actor
        use_proprio: Whether to use proprioceptive state information
        image_keys: List of image observation keys
        critic_network_kwargs: Configuration for critic networks
        policy_network_kwargs: Configuration for policy networks  
        policy_kwargs: Configuration for policy distribution
        critic_ensemble_size: Number of critics in ensemble
        discount: Discount factor for RL
        soft_target_update_rate: Tau for soft target updates
        target_entropy: Target entropy for SAC (if None, uses -action_dim)
        utd_ratio: Update-to-data ratio for training
    """

    # Input / output structure
    n_obs_steps: int = 1
    action_dim: int = 7

    # Normalization settings
    normalization_mapping: Dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            # Images often pre-normalized
            "VISUAL": NormalizationMode.IDENTITY,
            # Joint states normalized to [-1,1]
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.IDENTITY,
            # Actions normalized to [-1,1]
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # SERL Agent Configuration
    encoder_type: str = "resnet-pretrained"
    shared_encoder: bool = True
    use_proprio: bool = True
    image_keys: list = field(default_factory=lambda: ["image"])

    # Network architectures
    critic_network_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"hidden_dims": [256, 256]}
    )
    policy_network_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"hidden_dims": [256, 256]}
    )
    policy_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        }
    )

    # RL hyperparameters
    critic_ensemble_size: int = 2
    discount: float = 0.95
    soft_target_update_rate: float = 0.005
    target_entropy: float = None  # Will be set to -action_dim if None
    utd_ratio: int = 1

    # Training presets
    optimizer_lr: float = 3e-4
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()

        # Set target entropy if not provided
        if self.target_entropy is None:
            self.target_entropy = -self.action_dim

        # Validate configuration
        if self.action_dim <= 0:
            raise ValueError(
                f"action_dim must be positive, got {self.action_dim}")

        if (self.use_proprio and
                "observation.state" not in self.input_features):
            # We'll add this during validate_features if needed
            pass

    @classmethod
    def from_act_config(
        cls,
        act_config: PreTrainedConfig,
        encoder_type: str = "resnet-pretrained",
        shared_encoder: bool = True,
        critic_ensemble_size: int = 2,
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        utd_ratio: int = 1,
        **kwargs
    ) -> "SERLConfig":
        """
        Create SERLConfig from an existing ACT policy config.

        This preserves all the input/output features, normalization settings,
        and other environment-specific configurations from the ACT policy.

        Args:
            act_config: The ACT policy configuration to base this on
            encoder_type: SERL encoder type to use
            shared_encoder: Whether to share encoder between actor/critic
            critic_ensemble_size: Number of critics in ensemble
            discount: RL discount factor
            soft_target_update_rate: Soft target update rate
            utd_ratio: Update-to-data ratio
            **kwargs: Additional SERL-specific parameters

        Returns:
            SERLConfig instance with ACT features preserved
        """
        # Extract action dimension from ACT config
        action_feature = act_config.action_feature
        if action_feature is None:
            raise ValueError("ACT config must have an action feature defined")
        action_dim = action_feature.shape[0]

        # Determine image keys from ACT config
        image_keys = []
        for key in act_config.image_features.keys():
            # Extract image key from full feature name
            # e.g., "observation.images.main" -> "main"
            if key.startswith("observation.images."):
                image_key = key.split(".")[-1]
                image_keys.append(image_key)

        # Determine if we use proprioceptive state
        use_proprio = act_config.robot_state_feature is not None

        # Create SERL config with ACT settings
        serl_config = cls(
            # Copy basic settings
            n_obs_steps=act_config.n_obs_steps,
            action_dim=action_dim,
            device=act_config.device,

            # Copy normalization settings
            normalization_mapping=act_config.normalization_mapping.copy(),

            # SERL-specific settings
            encoder_type=encoder_type,
            shared_encoder=shared_encoder,
            use_proprio=use_proprio,
            image_keys=image_keys,
            critic_ensemble_size=critic_ensemble_size,
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,
            utd_ratio=utd_ratio,

            # Override with any additional kwargs
            **kwargs
        )

        # Copy input and output features directly
        serl_config.input_features = act_config.input_features.copy()
        serl_config.output_features = act_config.output_features.copy()

        return serl_config

    def validate_features(self) -> None:
        """Validate and setup input/output features for SERL policy."""

        # Ensure we have action output feature
        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim,),
            )
            self.output_features["action"] = action_feature

        # Ensure we have state input if use_proprio is True
        if (self.use_proprio and
                "observation.state" not in self.input_features):
            # We'll need to determine state dim from environment/dataset
            # For now, we'll set a placeholder that should be updated
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(7,),  # Common robot joint space
            )
            self.input_features["observation.state"] = state_feature

        # Ensure we have image features if image_keys are specified
        for image_key in self.image_keys:
            full_key = (f"observation.images.{image_key}"
                        if not image_key.startswith("observation")
                        else image_key)
            if full_key not in self.input_features:
                # Default camera resolution - should be updated
                # based on actual env
                image_feature = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, 224, 224),
                )
                self.input_features[full_key] = image_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        # SERL typically uses constant learning rate
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
