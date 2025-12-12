from dataclasses import dataclass, field

from flax import nnx

from .serializable import SerializableModule


@dataclass(kw_only=True)
class DistQNetworkLinear(SerializableModule):
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
    rngs_kwargs: dict[str, int]
    
    def __post_init__(self):
        self.reseed(self.rngs_kwargs)
    
    def reseed(self, rngs_kwargs: dict[str, int]):
        """Re-create model parameters using a dictionary of `nnx.Rngs` keyword arguments."""
        self.rngs_kwargs = rngs_kwargs
        rngs: nnx.Rngs = nnx.Rngs(**rngs_kwargs)
        
        # Organize dimensions so that the output of one is the input of another.
        dims = [self.din, *self.douts, self.num_actions * self.num_atoms]
        din_dout = list(zip(dims[:-1], dims[1:])) # (din, dout)
        self.layers = [
            nnx.Linear(in_features=in_features, out_features=out_features, rngs=rngs)
            for in_features, out_features in din_dout
        ]
    
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
