
# Updated for JAX 0.7.2: use Flax instead of deprecated stax
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

# Simple MLP with LeakyReLU activations
class MLP(nn.Module):
    features: Sequence[int]
    out_dim: int

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.leaky_relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


def CIGALE_emulator():
    output_cols = ['spire_250', 'spire_350', 'spire_500']
    model = MLP(features=[128, 128], out_dim=len(output_cols))
    return model



def CIGALE_emulator_kasia():
    output_cols = ['irac_i1', 'omegacam_g', 'omegacam_i', 'omegacam_r', 'omegacam_u', 'omegacam_z', 'spire_250', 'spire_350', 'spire_500']
    model = MLP(features=[128, 128, 128, 128], out_dim=len(output_cols))
    return model


def CIGALE_emulator_GEP():
    output_cols = [f'GEP{i}' for i in range(1, 24)]
    model = MLP(features=[128, 128, 128, 128], out_dim=len(output_cols))
    return model
