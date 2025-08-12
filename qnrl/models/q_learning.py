from dataclasses import dataclass, field

from flax import nnx
import jax.numpy as jnp

# See save/load example: https://github.com/google/flax/blob/main/examples/nnx_toy_examples/08_save_load_checkpoints.py

@dataclass(kw_only=True)
class QNetworkLinear(nnx.Module):
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

@dataclass(kw_only=True)
class DistQNetworkLinear(nnx.Module):
    """Distributional Q-learning network with linear layers.
    
    Creates a probability mass function (PMF) across the space of return-atoms for every action.
    
    Input:
        `(batch,din,)`: Input features must be 1-dimensional, or batched 2-dimensional.
    
    Output:
        `(batch,num_actions,num_atoms,)`: A probability mass function (PMF) across the space of return-atoms for every action.
    
    Arguments:
        din (int): din (int): Input feature dimension.
        num_actions (int): Number of discrete actions.
        num_atoms (int): Number of atoms for the return distribution.
        rngs (nnx.Rngs): JAX RNG.
        douts (list[int], optional): Intermediate (middle) layer output dimensions. Defaults to [].
    
    Reference:
        M. G. Bellemare, W. Dabney, and R. Munos, “A distributional perspective on reinforcement learning,” in Proceedings of the 34th international conference on machine learning, D. Precup and Y. W. Teh, Eds., in Proceedings of machine learning research, vol. 70. PMLR, Aug. 2017, pp. 449–458. doi: 10.48550/arXiv.1707.06887.
    """
    din: int
    num_actions: int
    num_atoms: int
    douts: list[int] = field(default_factory=list)

    def __init__(self,
        din: int,
        num_actions: int,
        num_atoms: int,
        *,
        rngs: nnx.Rngs,
        douts: list[int] = [],
        ):
        
        # Organize dimensions so that the output of one is the input of another.
        dims = [din, *douts, num_actions * num_atoms]
        din_dout = list(zip(dims[:-1], dims[1:])) # (din, dout)
        self.layers = [
            nnx.Linear(in_features=in_features, out_features=out_features, rngs=rngs)
            for in_features, out_features in din_dout
        ]
        self.din = din
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.douts = douts
    
    @nnx.jit
    def __call__(self, x):
        # Apply layer+ReLU to all except last layer.
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)
        # Generate return-atom distribution using last layer.
        x = self.layers[-1](x)
        x = x.reshape((-1, self.num_actions, self.num_atoms)) # (batch, action, atom)
        x = nnx.softmax(x, axis=-1) # Distribution over the atoms for every action.
        return x