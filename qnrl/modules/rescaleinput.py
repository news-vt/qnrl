from dataclasses import dataclass

from flax import nnx
import jax.numpy as jnp

from .serializable import SerializableModule

@dataclass(kw_only=True)
class RescaleInput(SerializableModule):
    """Divides input by scaling vector `x/scale`."""
    scale: list[float]
    
    def __init__(self, scale: list[float]):
        self.scale = scale
    
    @nnx.jit
    def __call__(self, x):
        x = x/jnp.array(self.scale)
        return x