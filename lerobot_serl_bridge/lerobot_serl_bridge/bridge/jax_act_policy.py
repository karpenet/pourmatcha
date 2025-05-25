"""
JAX/Flax implementation of ACT policy compatible with SERL.
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Optional, Tuple


class JAXACTPolicy(nn.Module):
    """JAX/Flax implementation of ACT policy compatible with SERL"""

    action_dim: int
    chunk_size: int = 100
    hidden_dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout_rate: float = 0.1
    use_vae: bool = True
    latent_dim: int = 32

    def setup(self):
        # Vision encoder (simplified - you may want to use a more
        # sophisticated encoder)
        self.vision_encoder = nn.Sequential([
            nn.Conv(32, kernel_size=(3, 3), strides=(2, 2)),
            nn.relu,
            nn.Conv(64, kernel_size=(3, 3), strides=(2, 2)),
            nn.relu,
            nn.Conv(128, kernel_size=(3, 3), strides=(2, 2)),
            nn.relu,
            nn.Conv(256, kernel_size=(3, 3), strides=(2, 2)),
            nn.relu,
            lambda x: x.reshape(x.shape[0], -1),  # Flatten
            nn.Dense(self.hidden_dim),
            nn.relu,
        ])

        # State encoder
        self.state_encoder = nn.Dense(self.hidden_dim)

        # Transformer encoder
        self.transformer_encoder = nn.Sequential([
            nn.SelfAttention(
                num_heads=self.n_heads,
                qkv_features=self.hidden_dim,
                dropout_rate=self.dropout_rate,
                deterministic=False
            ) for _ in range(self.n_layers)
        ])

        # VAE components
        if self.use_vae:
            self.vae_encoder = nn.Dense(self.latent_dim * 2)  # mu and log_var
            self.vae_decoder = nn.Dense(self.hidden_dim)

        # Action decoder
        self.action_decoder = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.action_dim * self.chunk_size),
        ])

    def encode_observations(
        self,
        observations: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Encode observations (images + state) into latent representation"""
        features = []

        # Encode images if present
        if "image" in observations:
            image_features = self.vision_encoder(observations["image"])
            features.append(image_features)

        # Encode state if present
        if "state" in observations:
            state_features = self.state_encoder(observations["state"])
            features.append(state_features)

        # Concatenate all features
        if len(features) > 1:
            encoded = jnp.concatenate(features, axis=-1)
        else:
            encoded = features[0]

        return encoded

    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        actions: Optional[jnp.ndarray] = None,
        training: bool = True
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """Forward pass through the ACT policy"""

        # Encode observations
        obs_encoded = self.encode_observations(observations)
        batch_size = obs_encoded.shape[0]

        # Add sequence dimension for transformer
        obs_encoded = obs_encoded[:, None, :]  # (batch, 1, hidden_dim)

        # If training and actions provided, add them to the sequence
        if training and actions is not None:
            # Reshape actions to sequence format
            action_seq = actions.reshape(batch_size, -1, self.action_dim)
            # Simple action embedding (could be more sophisticated)
            action_embedded = nn.Dense(self.hidden_dim)(action_seq)
            # Concatenate observation and action tokens
            sequence = jnp.concatenate([obs_encoded, action_embedded], axis=1)
        else:
            sequence = obs_encoded

        # Apply transformer encoder
        for layer in self.transformer_encoder:
            if isinstance(layer, nn.SelfAttention):
                sequence = layer(sequence, deterministic=not training)
            else:
                sequence = layer(sequence)

        # Extract observation encoding (first token)
        encoded_obs = sequence[:, 0, :]  # (batch, hidden_dim)

        # VAE processing
        vae_outputs = None
        if self.use_vae and training:
            vae_params = self.vae_encoder(encoded_obs)
            mu, log_var = jnp.split(vae_params, 2, axis=-1)

            # Reparameterization trick
            if training:
                key = self.make_rng('dropout')
                import jax
                eps = jax.random.normal(key, mu.shape)
                latent = mu + jnp.exp(0.5 * log_var) * eps
            else:
                latent = mu

            # Decode latent
            decoded_features = self.vae_decoder(latent)
            vae_outputs = (mu, log_var)
        else:
            decoded_features = encoded_obs
            vae_outputs = None

        # Generate action predictions
        action_pred = self.action_decoder(decoded_features)
        action_pred = action_pred.reshape(
            batch_size, self.chunk_size, self.action_dim)

        return action_pred, vae_outputs
