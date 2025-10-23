from dataclasses import dataclass, field

from flax import nnx

from .serializable import SerializableModule


@dataclass(kw_only=True)
class QNetworkLinear(SerializableModule):
    """Q-Learning network with linear layers.
    
    Input:
        `(batch,din,)`: Input features must be 1-dimensional, or batched 2-dimensional.
    
    Output:
        `(batch,num_actions,)`: Q-value estimate for every action.
    
    Arguments:
        din (int): Input feature dimension.
        num_actions (int): Number of discrete actions.
        rngs (nnx.Rngs): JAX RNG.
        douts (list[int], optional): Intermediate (middle) layer output dimensions. Defaults to [].
    """
    din: int
    num_actions: int
    rngs_kwargs: dict[str, int]
    douts: list[int] = field(default_factory=list)
    
    def __init__(self, din: int, num_actions: int, *, rngs_kwargs: dict[str, int], douts: list[int] = []):
        rngs: nnx.Rngs = nnx.Rngs(**rngs_kwargs)
        dims = [din, *douts, num_actions]
        din_dout = list(zip(dims[:-1], dims[1:]))
        self.layers = [
            nnx.Linear(in_features=in_features, out_features=out_features, rngs=rngs)
            for in_features, out_features in din_dout
        ]
        self.din = din
        self.num_actions = num_actions
        self.douts = douts
        self.rngs_kwargs = rngs_kwargs
    
    @nnx.jit
    def __call__(self, x):
        # Apply layer+ReLU to all except last layer.
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)
        # Apply last layer without any ReLU.
        x = self.layers[-1](x)
        return x