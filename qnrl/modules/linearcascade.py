from dataclasses import dataclass, field

from flax import nnx

from .serializable import SerializableModule, activation_function_dispatcher


@dataclass(kw_only=True)
class LinearCascade(SerializableModule):
    """Sequence of linear layers with activation functions after all but the last layer.
    
    Input:
        `(batch,in_features,)`: Input features must be 1-dimensional, or batched 2-dimensional.
    
    Output:
        `(batch,out_features,)`: Q-value estimate for every action.
    
    Arguments:
        in_features (int): Input feature dimension.
        out_features (int): Number of discrete actions.
        rngs (nnx.Rngs): JAX RNG.
        douts (list[int], optional): Intermediate (middle) layer output dimensions. Defaults to [].
    """
    in_features: int
    out_features: int
    rngs_kwargs: dict[str, int]
    douts: list[int] = field(default_factory=list)
    activation: str = 'relu'
    
    def __init__(self, in_features: int, out_features: int, *, rngs_kwargs: dict[str, int], douts: list[int] = [], activation: str = 'relu'):
        rngs: nnx.Rngs = nnx.Rngs(**rngs_kwargs)
        dims = [in_features, *douts, out_features]
        din_dout = list(zip(dims[:-1], dims[1:]))
        self.layers = [
            nnx.Linear(in_features=in_features, out_features=out_features, rngs=rngs)
            for in_features, out_features in din_dout
        ]
        self.in_features = in_features
        self.out_features = out_features
        self.douts = douts
        self.rngs_kwargs = rngs_kwargs
        self.activation = activation
        
        assert activation in activation_function_dispatcher, f"unsupported activation function {activation}"
    
    @nnx.jit
    def __call__(self, x):
        # Apply layer+ReLU to all except last layer.
        for layer in self.layers[:-1]:
            x = layer(x)
            x = activation_function_dispatcher[self.activation](x)
        # Apply last layer without any ReLU.
        x = self.layers[-1](x)
        return x