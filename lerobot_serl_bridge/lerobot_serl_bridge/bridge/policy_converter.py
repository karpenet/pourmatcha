"""
Policy conversion utilities between PyTorch and JAX formats.
"""

import os
import logging
from typing import Dict, Tuple

from flax.training import checkpoints

# LeRobot imports
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig

from .jax_act_policy import JAXACTPolicy
from ..training.config import TrainingConfig

logger = logging.getLogger(__name__)


class PolicyConverter:
    """Converts policies between PyTorch and JAX formats"""

    @staticmethod
    def pytorch_to_jax(
        pytorch_policy: ACTPolicy,
        config: TrainingConfig
    ) -> JAXACTPolicy:
        """Convert PyTorch ACT policy to JAX format"""
        logger.info("Converting PyTorch ACT policy to JAX format...")

        # Extract architecture parameters from PyTorch config
        act_config = pytorch_policy.config

        # Create JAX policy with matching architecture
        jax_policy = JAXACTPolicy(
            action_dim=config.action_dim,
            chunk_size=act_config.chunk_size,
            hidden_dim=act_config.hidden_dim,
            n_layers=act_config.n_encoder_layers,
            n_heads=act_config.n_heads,
            dropout_rate=act_config.dropout,
            use_vae=act_config.use_vae,
            latent_dim=act_config.latent_dim if act_config.use_vae else 32,
        )

        logger.info("JAX ACT policy created successfully")
        return jax_policy

    @staticmethod
    def jax_to_pytorch(
        jax_params: Dict,
        original_config: ACTConfig,
        dataset_stats: Dict
    ) -> ACTPolicy:
        """Convert trained JAX parameters back to PyTorch ACT policy"""
        logger.info("Converting JAX parameters back to PyTorch format...")

        # Create new PyTorch policy
        pytorch_policy = ACTPolicy(original_config, dataset_stats)

        # TODO: Implement parameter mapping
        # This requires careful mapping between JAX and PyTorch parameter
        # structures
        logger.warning(
            "Parameter conversion from JAX to PyTorch not yet implemented")
        logger.info("Returning original PyTorch policy structure")

        return pytorch_policy

    @staticmethod
    def save_jax_policy(
        params: Dict,
        policy_def: JAXACTPolicy,
        save_path: str
    ):
        """Save JAX policy parameters"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save using Flax checkpointing
        checkpoints.save_checkpoint(
            save_path,
            target={"params": params, "policy_def": policy_def},
            step=0,
            overwrite=True
        )
        logger.info(f"JAX policy saved to {save_path}")

    @staticmethod
    def load_jax_policy(load_path: str) -> Tuple[Dict, JAXACTPolicy]:
        """Load JAX policy parameters"""
        checkpoint = checkpoints.restore_checkpoint(load_path, target=None)
        return checkpoint["params"], checkpoint["policy_def"]
