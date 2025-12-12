from dataclasses import dataclass, field

from flax import nnx

from .serializable import SerializableModule


@dataclass(kw_only=True)
class Linear(SerializableModule):
    """Serializable variant of `nnx.Linear`.
    
    This class is a serializable wrapper around `nnx.Linear`. The linear layer object can be access via the variable `self.linear`.
    """
    in_features: int
    out_features: int
    rngs_kwargs: dict[str, int]
    
    def __post_init__(self):
        self.reseed(self.rngs_kwargs)
    
    def reseed(self, rngs_kwargs: dict[str, int]):
        """Re-create model parameters using a dictionary of `nnx.Rngs` keyword arguments."""
        self.rngs_kwargs = rngs_kwargs
        rngs: nnx.Rngs = nnx.Rngs(**rngs_kwargs)
        self.linear = nnx.Linear(in_features=self.in_features, out_features=self.out_features, rngs=rngs)
    
    @nnx.jit
    def __call__(self, x):
        return self.linear(x)