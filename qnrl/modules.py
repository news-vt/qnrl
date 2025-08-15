from dataclasses import dataclass, field
from functools import partial, reduce
from typing import Callable, Any

import jax
from flax import nnx
import jax.numpy as jnp
import pennylane as qml
from mashumaro import DataClassDictMixin
from mashumaro.config import BaseConfig
from mashumaro.types import Discriminator

# See save/load example: https://github.com/google/flax/blob/main/examples/nnx_toy_examples/08_save_load_checkpoints.py


@dataclass(kw_only=True)
class SerializableModule(nnx.Module, DataClassDictMixin):
    module_id: str = field(default='SerializableModule', init=False)
    
    def __post_init__(self):
        self.module_id = self.__class__.__name__
    
    class Config(BaseConfig):
        discriminator = Discriminator(
            field="module_id",
            include_subtypes=True,
        )
    
    @nnx.jit
    def __call__(self, x):
        """Default call function returns the input unchanged."""
        return x


@dataclass(kw_only=True)
class Sequential(SerializableModule):
    """Sequential `nnx.Module` using either a `list[...]` or `dict[str, ...]` of modules.
    
    Always behaves like a list of elements, where the indices are either `int` (if layers are `list`) or `str` (if layers are `dict`).
    
    BEWARE of a quirk with `nnx.jit` where dictionary keys are automatically sorted. As a result, when passing a `dict[str, ...]` the key insertion order is not guaranteed on the `self.layers` member after calling `__call__()`. To remedy this, always use `__iter__()` to iterate over the key list. The key order will be preserved when serializing via `to_dict()`.
    
    If a `list[...]` is provided the PyTree will look like:
    >>> {
            'layers': {
                0: { ... },
                1: { ... },
                ...
            },
        }
    
    If a `dict[str, ...]` is provided, the original key order will be preserved via `__iter__()`, with the PyTree:
    >>> {
            'layers': {
                'key-0': { ... },
                'key-1': { ... },
                ...
            },
        }
    """
    module_id: str = field(default='Sequential', init=False)
    layers: list[SerializableModule] | dict[str, SerializableModule]
    key_order: list[int] | list[str] # Private layer key ordering.
    
    def __init__(self, layers: list[SerializableModule] | dict[str, SerializableModule], key_order: list[int] | list[str] | None = None):
        self.layers = layers
        
        # Preserve the original layer order, because of a bug with `nnx.jit` automatically sorting dictionary keys.
        if key_order is None:
            self.key_order = list(self.layers.keys()) if isinstance(self.layers, dict) else list(range(len(self.layers)))
        else:
            self.key_order = key_order
    
    def __iter__(self):
        yield from (self.layers[key] for key in self.key_order)
    
    def __getitem__(self, key):
        return self.layers[key]
    
    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        # Preserve original input key order due to quirk of `nnx.jit`.
        if isinstance(d['layers'], dict):
            d['layers'] = {key: d['layers'][key] for key in self.key_order}
        return d
    
    @nnx.jit
    def __call__(self, x):
        for key in self.key_order:
            x = self.layers[key](x)
        return x


@dataclass(kw_only=True)
class RescaleInput(SerializableModule):
    module_id: str = field(default='RescaleInput', init=False)
    scale: list[float]
    
    def __init__(self, scale: list[float]):
        self.scale = scale
    
    @nnx.jit
    def __call__(self, x):
        x = x/jnp.array(self.scale)
        return x


@dataclass(kw_only=True)
class QuantumModule(SerializableModule):
    module_id: str = field(default='QuantumModule', init=False)
    rngs_kwargs: dict[str, int]
    device_name: str = field(default="default.qubit")
    diff_method: str | None = None
    shots: int | None = None
    
    def __init__(self,
        *,
        rngs_kwargs: dict[str, int],
        device_name: str = "default.qubit",
        diff_method: str | None = None,
        shots: int | None = None,
        ):
        self.rngs_kwargs = rngs_kwargs
        self.rngs: nnx.Rngs = nnx.Rngs(**rngs_kwargs)
        self.device_name = device_name
        self.diff_method = diff_method
        self.shots = shots
        
        self.circuit_kwargs = dict()
    
    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        d.pop('circuit_kwargs', None) # Remove circuit keyword arguments.
        return d
    
    @nnx.jit
    def __call__(self, x):
        key = self.rngs.default() # Generate new random key for circuit measurement.
        # print(f"{x.shape = }")
        x = self.run_circuit(
            key=key,
            device_name=self.device_name,
            diff_method=self.diff_method,
            shots=self.shots,
            inputs=x,
            **self.circuit_kwargs,
        ) # List of measurements, where each measurement is batched.
        # Reverse the shape to get back into a form acceptable for future layers.
        x = jnp.array(x).T # (batch,*,)
        return x
    
    @staticmethod
    def circuit(
        inputs: jax.Array, # (num_wires,1)
        **kwargs,
        ):
        raise NotImplementedError

    @staticmethod
    @partial(nnx.jit, static_argnames=['device_name', 'shots', 'diff_method', 'num_wires', 'num_layers'])
    def run_circuit(
        # All device and QNode parameters.
        key: jax.Array,
        device_name: str,
        shots: int | None,
        diff_method: str | None,
        # All circuit-specific arguments go below.
        inputs: jax.Array, # (num_wires,1)
        weights: jax.Array, # (num_layers,num_wires,3)
        num_wires: int,
        num_layers: int,
        ):
        device = qml.device(name=device_name, shots=shots, seed=key)
        qnode = qml.QNode(func=QuantumQNetworkLinear.circuit, device=device, interface="jax", diff_method=diff_method)
        qnode_vmap = nnx.vmap(qnode, in_axes=(0, None, None, None), out_axes=0)
        return qnode_vmap(
            inputs=inputs,
            weights=weights,
            num_wires=num_wires,
            num_layers=num_layers,
        )

# https://discuss.pennylane.ai/t/trouble-calling-qjit-compiled-qnode-inside-a-nnx-jit-compiled-flax-nnx-module/7853
# https://discuss.pennylane.ai/t/error-combining-jax-with-multiple-devices/4934
# https://discuss.pennylane.ai/t/facing-issues-with-jax-jitting-the-optimization-loop/4274
@dataclass(kw_only=True)
class QuantumQNetworkLinear(SerializableModule):
    module_id: str = field(default='QuantumQNetworkLinear', init=False)
    din: int
    num_actions: int
    num_wires: int
    num_layers: int
    rngs_kwargs: dict[str, int]

    def __init__(self, din: int, num_actions: int, num_wires: int, num_layers: int, *, rngs_kwargs: dict[str, int]):
        rngs: nnx.Rngs = nnx.Rngs(**rngs_kwargs)

        self.rngs_kwargs = rngs_kwargs
        self.din = din
        self.num_actions = num_actions
        self.num_wires = num_wires
        self.num_layers = num_layers
        
        self.circ_inputs_shape = (num_wires,1,) # Each qubit has (Rx,).
        self.circ_weights_shape = (num_layers+1,num_wires,3,) # Each qubit has (Rx, Ry, Rz).
        self.circ_outputs_shape = num_wires
        
        self.layer_project_input = nnx.Linear(in_features=din, out_features=reduce(lambda x,y: x*y, self.circ_inputs_shape), rngs=rngs)
        self.layer_project_output = nnx.Linear(in_features=num_wires, out_features=num_actions, rngs=rngs)
        
        key = rngs.default()
        self.circ = self.make_circuit(num_wires=num_wires, num_layers=num_layers, shots=100, key=key)
        # 
        # self.circuit_key = rngs.default()
        #####
        self.circ_weights = nnx.Param(jax.random.uniform(rngs.params(), shape=self.circ_weights_shape, minval=0, maxval=jnp.pi))
    
    @nnx.jit
    def __call__(self, x):
        x = self.layer_project_input(x)
        # Input must be batched for input into the circuit.
        x = jnp.reshape(x, (-1, *self.circ_inputs_shape,))
        # Output of circuit will be reverse-shaped list with shape `(num_wires,batch,)` due to how batching is handled with pennylane measurements.
        x = self.circ(x, self.circ_weights) # list of `num_wires`, where each index has shape `batch`.
        # Reverse the shape to get back into a form acceptable for future layers.
        x = jnp.array(x).T # (batch,num_wires,)
        x = self.layer_project_output(x)
        return x
    
    @staticmethod
    def circuit(
        inputs: jax.Array, # (num_wires,1)
        weights: jax.Array, # (num_layers,num_wires,3)
        num_wires: int,
        num_layers: int,
        ):
        """Variational and encoding circuit.
        
        Batching is not natively supported. To support batching, use `jax.vmap`.
        
        Args:
            inputs (jax.Array): Input array of shape `(num_wires,3)`.
            num_wires (int): Number of wires.
            num_layers (int): Number of layers.
        
        -- RX(in[w=0,0]) -- RY(in[w=0,1]) -- RZ(in[w=0,2]) --
        -- RX(in[w=1,0]) -- RY(in[w=1,1]) -- RZ(in[w=1,2]) --
        -- ...
        """
        
        def var(phi: jax.Array, wires: list[int]):
            """Variational layer.

            Args:
                phi (jax.Array): Variational parameters for the layer of shape `(num_wires,3)`.
                wires (list[int]): List of wires.
            """
            for w in wires:
                qml.RX(phi=phi[w,0], wires=w)
                qml.RY(phi=phi[w,1], wires=w)
                qml.RZ(phi=phi[w,2], wires=w)
        
        def circ(wires: list[int]):
            """Circular entanglement layer using `CZ`.

            Args:
                wires (list[int]): List of wires.
            """
            for w0, w1 in zip(wires, wires[1:]):
                qml.CZ(wires=[w0, w1])
            if len(wires) != 2:
                qml.CZ(wires=[wires[0], wires[-1]])
        
        def enc(phi: jax.Array, wires: list[int]):
            """Encoding layer using `RX`.

            Args:
                phi (jax.Array): Encoding parameters for the layer of shape `(num_wires,1)`.
                wires (list[int]): _description_
            """
            for w in wires:
                qml.RX(phi=phi[w,0], wires=w)
        
        # Create list of wire indices.
        wires = list(range(num_wires))
        
        # Cascaded variational, circular entanglement, and encoding layers.
        for l in range(num_layers):
            var(phi=weights[l], wires=wires)
            circ(wires=wires)
            enc(phi=inputs, wires=wires)
        # Final variational layer at the end.
        var(phi=weights[-1], wires=wires)
        
        return [qml.expval(qml.PauliZ(wires=w)) for w in wires]
    
    @classmethod
    # @partial(jax.jit, static_argnames=['cls', 'num_wires', 'num_layers', 'shots'])
    def make_circuit(cls, num_wires: int, num_layers: int, shots: int, *, key: jax.Array):
        device = qml.device(name="default.qubit", shots=shots, seed=key)
        qnode = qml.QNode(func=lambda inputs, weights: cls.circuit(inputs, weights, num_wires, num_layers), device=device, interface="jax-jit")
        # return qnode
        return jax.vmap(qnode, in_axes=(0, None), out_axes=0) # (inputs, weights), Input is to be batched, weights is not.
        # return jax.vmap(qnode, in_axes=(0, None)) # (inputs, weights), Input is to be batched, weights is not.
    
    # TODO method that is JIT that calls the circuit and returns result.
    # @classmethod
    # # @partial(jax.jit, static_argnames=['cls', 'num_wires', 'num_layers', 'shots'])
    # def run_circuit(cls, )


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
    module_id: str = field(default='QNetworkLinear', init=False)
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
    module_id: str = field(default='DistQNetworkLinear', init=False)
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