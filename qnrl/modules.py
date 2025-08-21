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

def module_tagger(cls):
    """Tag serializable modules by their class name."""
    return cls.__name__


@dataclass(kw_only=True)
class SerializableModule(nnx.Module, DataClassDictMixin):
    """Base class for Flax NNX modules that can be serialized to dictionaries.
    
    Variants of this class are distinguished via the `"type"` field that is auto-generated upon serialization. By default, this `"type"` field is the class string name `cls.__name__`. See the `module_tagger` function at module scope.
    """
    class Config(BaseConfig):
        discriminator = Discriminator(
            field="type",
            include_subtypes=True,
            variant_tagger_fn=module_tagger,
        )
    
    @nnx.jit
    def __call__(self, x):
        """Default call function returns the input unchanged."""
        return x
    
    def __pre_serialize__(self):
        return self
    
    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        # Re-build serialization dictionary by creating a `type` field as the first key.
        d = {
            'type': module_tagger(type(self)), # Discriminator field.
            **d,
        }
        return d


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
        # Get serialization of parent.
        d = super().__post_serialize__(d)
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
    """Divides input by scaling vector `x/scale`."""
    scale: list[float]
    
    def __init__(self, scale: list[float]):
        self.scale = scale
    
    @nnx.jit
    def __call__(self, x):
        x = x/jnp.array(self.scale)
        return x


# https://discuss.pennylane.ai/t/trouble-calling-qjit-compiled-qnode-inside-a-nnx-jit-compiled-flax-nnx-module/7853
# https://discuss.pennylane.ai/t/error-combining-jax-with-multiple-devices/4934
# https://discuss.pennylane.ai/t/facing-issues-with-jax-jitting-the-optimization-loop/4274
@dataclass(kw_only=True)
class QuantumQNetworkLinear(SerializableModule):
    din: int
    num_actions: int
    num_wires: int
    num_layers: int
    rngs_kwargs: dict[str, int]
    device_name: str = "default.qubit"
    diff_method: str | None = None
    shots: int | None = None

    def __init__(self,
        din: int,
        num_actions: int,
        num_wires: int,
        num_layers: int,
        *,
        rngs_kwargs: dict[str, int],
        device_name: str = "default.qubit",
        diff_method: str | None = None,
        shots: int | None = None,
        ):
        self.rngs_kwargs = rngs_kwargs
        self.rngs: nnx.Rngs = nnx.Rngs(**rngs_kwargs)
        self.din = din
        self.num_actions = num_actions
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.device_name = device_name
        self.diff_method = diff_method
        self.shots = shots
        
        self.circ_inputs_shape = (num_wires,1,) # Each qubit has (Rx,).
        self.circ_weights_shape = (num_layers+1,num_wires,3,) # Each qubit has (Rx, Ry, Rz).
        self.circ_outputs_shape = num_wires
        
        self.layer_project_input = nnx.Linear(in_features=din, out_features=reduce(lambda x,y: x*y, self.circ_inputs_shape), rngs=self.rngs)
        self.layer_project_output = nnx.Linear(in_features=num_wires, out_features=num_actions, rngs=self.rngs)
        
        self.circ_weights = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_shape, minval=0, maxval=jnp.pi))
    
    @nnx.jit
    def __call__(self, x):
        x = self.layer_project_input(x)
        # Input must be batched for input into the circuit.
        x = jnp.reshape(x, (-1, *self.circ_inputs_shape,))
        # Output of circuit will be reverse-shaped list with shape `(num_wires,batch,)` due to how batching is handled with pennylane measurements.
        key = self.rngs.default() # Generate new random key for circuit measurement.
        x = self.run_circuit(
            key=key,
            device_name=self.device_name,
            diff_method=self.diff_method,
            shots=self.shots,
            inputs=x,
            weights=self.circ_weights,
            num_wires=self.num_wires,
            num_layers=self.num_layers,
        )
        # # Reverse the shape to get back into a form acceptable for future layers.
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

    # TODO method that is JIT that calls the circuit and returns result.
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
        qnode_vmap = nnx.vmap(
            f=qnode,
            in_axes=(0, None, None, None), # (inputs, weights, num_wires, num_layers)
            out_axes=0, # inputs
        )
        return qnode_vmap(
            inputs=inputs,
            weights=weights,
            num_wires=num_wires,
            num_layers=num_layers,
        )


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

    def __init__(self,
        din: int,
        num_actions: int,
        num_atoms: int,
        *,
        rngs_kwargs: dict[str, int],
        douts: list[int] = [],
        ):
        rngs: nnx.Rngs = nnx.Rngs(**rngs_kwargs)
        
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
        self.rngs_kwargs = rngs_kwargs
    
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










########################################
# MARK: QNRL

@dataclass(kw_only=True)
class QuantumNativeQNetwork(SerializableModule):
    """Quantum-native (distributional) Q-value network."""
    din: int
    # n_policies: int
    # """Number of policies to optimize (will require `ceil(log2(n))` qubits)."""
    n_actions: int
    """Number of actions (will require `ceil(log2(n))` qubits)."""
    n_returns: int
    """Number of return atoms (will require `ceil(log2(n))` qubits)."""
    z_min: float
    z_max: float
    m_moment: int
    """m-th moment of the return atoms distribution (`E[Z^m]`)."""
    n_reuploads: int
    """n-th power of the expectation over the return atoms distribution (`E[Z]^n`)."""
    n_layers: int
    """Number of re-upload layers to use for the return-atom distribution unitary."""
    rngs_kwargs: dict[str, int]
    device_name: str = "default.qubit"
    diff_method: str | None = None
    shots: int | None = None
    
    def __post_init__(self):
        self.rngs: nnx.Rngs = nnx.Rngs(**self.rngs_kwargs)
        
        # Compute qubit count based on number of policies, actions, and return atoms.
        # self.d_policies = int(jnp.ceil(jnp.log2(self.n_policies))) # Qubit count for policies to optimize.
        self.d_actions = int(jnp.ceil(jnp.log2(self.n_actions))) # Qubit count for actions.
        self.d_returns = int(jnp.ceil(jnp.log2(self.n_returns))) # Qubit count for return atoms.
        
        # Collect indices of actions based on the total number of actions that can be represented by the number of qubits on the action register.
        self.all_action_idx = list(range(2**self.d_actions)) # All actions represented in the action qubit register.
        self.active_action_idx = list(range(self.n_actions)) # Indices of actions that are being used on the action qubit register.
        self.inactive_action_idx = list(set(self.all_action_idx) - set(self.active_action_idx)) # Indices of actions that are not being used on the action qubit register.
        
        # Create list of return atoms.
        self.atoms = jnp.linspace(start=self.z_min, stop=self.z_max, num=self.n_returns)
        # Create non-linear trace non-increasing Kraus matrix of returns atoms.
        atoms_values = nnx.softmax(self.atoms) # Normalize to probability distribution.
        atoms_values = jnp.sqrt(atoms_values * self.n_returns) # Ensure that L2-norm of quantum state amplitude will produce the atom probability (i.e., $|amp|^2 = prob$).
        atoms_values = atoms_values / atoms_values.max() # Ensure maximum value is 1.
        if self.n_returns != 2**self.d_returns: # Pad with zeros if return atoms shape is not a power of 2.
            atoms_values = jnp.pad(atoms_values, (0,2**self.d_returns - self.n_returns), mode='constant')
        self.U_atoms = jnp.diag(atoms_values)
        
        # TODO returns probability matrix with shape (n_policy,n_action,n_return).
        # self.returns_probs_matrix = nnx.Param(
        #     nnx.softmax( # Normalize to make it a probability vector (sums to 1).
        #         x=jax.random.normal( # Atom probabilities sampled from normal distribution.
        #             key=self.rngs.params(),
        #             # shape=(self.n_policies, self.n_actions, self.n_returns),
        #             shape=(self.n_actions, self.n_returns),
        #         ),
        #         axis=-1, # Softmax over `n_returns` dimension.
        #     )
        # )
        
        # Weights for return-atom distribution generation.
        # Every action gets their own return-atom distribution unitary.
        self.circ_weights_shape = (self.n_actions,self.n_layers,self.d_returns,3,) # Each qubit has (Rx, Ry, Rz).
        self.circ_weights = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_shape, minval=0, maxval=jnp.pi))
        
        # Inputs to circuit will be encoded onto the return-atom wires.
        self.circ_inputs_shape = (self.d_returns,1,) # Each qubit has (Rx,).
        self.layer_project_input = nnx.Linear(in_features=self.din, out_features=reduce(lambda x,y: x*y, self.circ_inputs_shape), rngs=self.rngs)
    
    @nnx.jit
    def __call__(self, x): # (batch,din,)
        # Project input into shape that is compatible with the quantum circuit.
        x = self.layer_project_input(x)
        # Input must be batched for input into the circuit.
        x = jnp.reshape(x, (-1, *self.circ_inputs_shape,))
        # Run quantum circuit.
        key = self.rngs.default() # Generate new random key for circuit measurement.
        x = self.run_circuit( # (batch,2**d_actions,)
            # All device and QNode parameters.
            key=key,
            device_name=self.device_name,
            shots=self.shots,
            diff_method=self.diff_method,
            # Circuit arguments.
            inputs=x,
            weights=self.circ_weights,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            m_moment=self.m_moment,
            n_reuploads=self.n_reuploads,
            n_layers=self.n_layers,
            U_atoms=self.U_atoms,
        )
        # Filter circuit output based on number of "active" actions.
        # If desired number of actions is not a multiple of 2, then filter out the excess actions on the qubit register (i.e., more actions can be represented on the register and we don't care about them).
        x = x[:,jnp.array(self.active_action_idx, dtype=int)] # (batch,n_actions,)
        # Ensure probabilities are normalized after the filter.
        x = x / x.sum(axis=-1, keepdims=True)
        return x
    
    @staticmethod
    def circuit(
        inputs: jax.Array, # (d_returns,1,)
        weights: jax.Array, # (n_actions,n_layers,d_returns,3,)
        n_actions: int,
        d_actions: int,
        d_returns: int,
        m_moment: int,
        n_reuploads: int, # >= 1.
        n_layers: int,
        U_atoms: jax.Array,
        ):
        
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
        
        def reset(wires: list[int]):
            """Measure the given qubits to collapse their state, then reset to zero."""
            for w in wires:
                qml.measure(wires=w, reset=True)
        
        def get_ops_for_returns_distribution(
            inputs: jax.Array, # (d_returns,1,)
            weights: jax.Array, # (n_layers,d_returns,3,)
            n_layers: int,
            wires: list[int],
            ):
            """Return-atom distribution unitary.
            
            Layered variational, circular-entanglement, and encoding circuit.
            """
            for l in range(n_layers):
                var(phi=weights[l], wires=wires)
                circ(wires=wires)
                enc(phi=inputs, wires=wires)
            var(phi=weights[-1], wires=wires) # Final variational layer at the end.
        
        def prepare_returns_probability_distribution(
            inputs: jax.Array, # (d_returns,1,)
            weights: jax.Array, # (n_actions,n_layers,d_returns,3,)
            n_actions: int,
            n_layers: int,
            wires_actions: list[int],
            wires_returns: list[int],
            ):
            ops = [
                qml.prod(get_ops_for_returns_distribution)(
                    inputs=inputs,
                    weights=weights[a],
                    n_layers=n_layers,
                    wires=wires_returns,
                )
                for a in range(n_actions)
            ]
            qml.Select(ops, control=wires_actions)
        
        def prepare_m_moment(
            m_moment: int,
            U_atoms: jax.Array,
            wires_returns: list[int],
            ):
            # Apply the (non-unitary) return-atom-value operation.
            # qml.QubitUnitary(U_returns_atoms, wires=wires_returns, unitary_check=False) # We know it's not unitary, so don't check.
            for _ in range(m_moment):
                qml.QubitUnitary(U_atoms, wires=wires_returns, unitary_check=False) # We know it's not unitary, so don't check.
        
        # Create wires.
        num_wires = d_actions + d_returns
        wirelist = list(range(num_wires)) # All wires.
        wiredict = { # Wires separated into blocks for each section.
            'actions': wirelist[:d_actions],
            'returns': wirelist[d_actions:]
        }
        
        # Equal superposition of all actions for each policy.
        # Use `StatePrep` here because there may be fewer "active" actions than the total number of actions supported by the number of qubits in the action register. This ensures that all "inactive" actions have 0 amplitude.
        qml.StatePrep(jnp.ones(shape=(n_actions,)), wires=wiredict['actions'], normalize=True, pad_with=0)
        
        # Prepare the n-th power of the m-th moment of the returns distribution.
        for _ in range(n_reuploads):
            # Prepare returns probability distribution for every policy.
            prepare_returns_probability_distribution(
                inputs=inputs,
                weights=weights,
                n_actions=n_actions,
                n_layers=n_layers,
                wires_actions=wiredict['actions'],
                wires_returns=wiredict['returns'],
            )
            # Prepare the m-th moment of the returns distribution.
            prepare_m_moment(
                m_moment=m_moment,
                U_atoms=U_atoms,
                wires_returns=wiredict['returns'],
            )
            # Measure the return-atom qubits to collapse their state, then reset to zero.
            reset(wires=wiredict['returns'])
        
        return qml.probs(wires=wiredict['actions'])
        # return (
        #     qml.probs(wires=wiredict['actions']),
        #     qml.probs(wires=wiredict['returns']),
        # )

    @staticmethod
    @partial(nnx.jit, static_argnames=[
        # Device and QNode parameters.
        'device_name', 'shots', 'diff_method',
        # Circuit parameters.
        'n_actions',
        'd_actions',
        'd_returns',
        'm_moment',
        'n_reuploads',
        'n_layers',
        'U_atoms',
    ])
    def run_circuit(
        # All device and QNode parameters.
        key: jax.Array,
        device_name: str,
        shots: int | None,
        diff_method: str | None,
        # Circuit arguments.
        inputs: jax.Array, # (d_returns,1,)
        weights: jax.Array, # (n_actions,n_layers,d_returns,3,)
        n_actions: int,
        d_actions: int,
        d_returns: int,
        m_moment: int,
        n_reuploads: int, # >= 1.
        n_layers: int,
        U_atoms: jax.Array,
        ):
        device = qml.device(name=device_name, shots=shots, seed=key)
        qnode = qml.QNode(func=QuantumNativeQNetwork.circuit, device=device, interface="jax", diff_method=diff_method)
        qnode_vmap = nnx.vmap(
            f=qnode,
            in_axes=(0, *([None]*8)), # (inputs, weights, n_actions, d_actions, d_returns, m_moment, n_reuploads, n_layers, U_atoms)
            out_axes=0, # inputs
        )
        return qnode_vmap(
            inputs=inputs,
            weights=weights,
            n_actions=n_actions,
            d_actions=d_actions,
            d_returns=d_returns,
            m_moment=m_moment,
            n_reuploads=n_reuploads,
            n_layers=n_layers,
            U_atoms=U_atoms,
        )




#############
#############
#############
#############
#############



@dataclass(kw_only=True)
class QuantumNativeQNetworkCascade(SerializableModule):
    """Cascaded variant of quantum-native (distributional) Q-value network to prevent memory explosion for high values of `n_reuploads`."""
    din: int
    # n_policies: int
    # """Number of policies to optimize (will require `ceil(log2(n))` qubits)."""
    n_actions: int
    """Number of actions (will require `ceil(log2(n))` qubits)."""
    n_returns: int
    """Number of return atoms (will require `ceil(log2(n))` qubits)."""
    z_min: float
    z_max: float
    m_moment: int
    """m-th moment of the return atoms distribution (`E[Z^m]`)."""
    n_reuploads: int
    """n-th power of the expectation over the return atoms distribution (`E[Z]^n`)."""
    n_layers: int
    """Number of re-upload layers to use for the return-atom distribution unitary."""
    rngs_kwargs: dict[str, int]
    device_name: str = "default.qubit"
    diff_method: str | None = None
    shots: int | None = None
    
    def __post_init__(self):
        self.rngs: nnx.Rngs = nnx.Rngs(**self.rngs_kwargs)
        
        # Compute qubit count based on number of policies, actions, and return atoms.
        # self.d_policies = int(jnp.ceil(jnp.log2(self.n_policies))) # Qubit count for policies to optimize.
        self.d_actions = int(jnp.ceil(jnp.log2(self.n_actions))) # Qubit count for actions.
        self.d_returns = int(jnp.ceil(jnp.log2(self.n_returns))) # Qubit count for return atoms.
        
        # Collect indices of actions based on the total number of actions that can be represented by the number of qubits on the action register.
        self.all_action_idx = list(range(2**self.d_actions)) # All actions represented in the action qubit register.
        self.active_action_idx = list(range(self.n_actions)) # Indices of actions that are being used on the action qubit register.
        self.inactive_action_idx = list(set(self.all_action_idx) - set(self.active_action_idx)) # Indices of actions that are not being used on the action qubit register.
        
        # Create list of return atoms.
        self.atoms = jnp.linspace(start=self.z_min, stop=self.z_max, num=self.n_returns)
        # Create non-linear trace non-increasing Kraus matrix of returns atoms.
        atoms_values = nnx.softmax(self.atoms) # Normalize to probability distribution.
        atoms_values = jnp.sqrt(atoms_values * self.n_returns) # Ensure that L2-norm of quantum state amplitude will produce the atom probability (i.e., $|amp|^2 = prob$).
        atoms_values = atoms_values / atoms_values.max() # Ensure maximum value is 1.
        if self.n_returns != 2**self.d_returns: # Pad with zeros if return atoms shape is not a power of 2.
            atoms_values = jnp.pad(atoms_values, (0,2**self.d_returns - self.n_returns), mode='constant')
        self.U_atoms = jnp.diag(atoms_values)
        
        # TODO returns probability matrix with shape (n_policy,n_action,n_return).
        # self.returns_probs_matrix = nnx.Param(
        #     nnx.softmax( # Normalize to make it a probability vector (sums to 1).
        #         x=jax.random.normal( # Atom probabilities sampled from normal distribution.
        #             key=self.rngs.params(),
        #             # shape=(self.n_policies, self.n_actions, self.n_returns),
        #             shape=(self.n_actions, self.n_returns),
        #         ),
        #         axis=-1, # Softmax over `n_returns` dimension.
        #     )
        # )
        
        # Weights for return-atom distribution generation.
        # Every action gets their own return-atom distribution unitary.
        self.circ_weights_shape = (self.n_actions,self.n_layers,self.d_returns,3,) # Each qubit has (Rx, Ry, Rz).
        self.circ_weights = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_shape, minval=0, maxval=jnp.pi))
        ####
        # self.circ_weights_base = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_shape, minval=0, maxval=jnp.pi))
        # self.circ_weights = [
        #     self.circ_weights_base for _ in range(self.n_reuploads)
        # ] # Same parameter for each reupload layer.
        # # self.circ_weights = [
        # #     nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_shape, minval=0, maxval=jnp.pi))
        # #     for _ in range(self.n_reuploads)
        # # ] # (n_reuploads,n_actions,n_layers,d_returns,3,)
        
        
        # Inputs to circuit will be encoded onto the return-atom wires.
        self.circ_inputs_shape = (self.d_returns,1,) # Each qubit has (Rx,).
        self.layer_project_input = nnx.Linear(in_features=self.din, out_features=reduce(lambda x,y: x*y, self.circ_inputs_shape), rngs=self.rngs)
    
    @nnx.jit
    def __call__(self, x): # (batch,din,)
        # Project input into shape that is compatible with the quantum circuit.
        x = self.layer_project_input(x)
        # Input must be batched for input into the circuit.
        x = jnp.reshape(x, (-1, *self.circ_inputs_shape,)) # (batch,d_returns,1,)
        # Prepare the n-th power of the m-th moment of the returns distribution.
        key = self.rngs.default() # Generate new random key for circuit measurement.
        probs_actions = jnp.ones(shape=(x.shape[0], self.n_actions,), dtype='float64')
        for _ in range(self.n_reuploads):
            # jax.debug.print("[QuantumNativeQNetworkCascade] Upload {n}: start, {probs_actions}", n=n, probs_actions=probs_actions)
            # Run quantum circuit.
            key, subkey = jax.random.split(key=key)
            probs_actions = self.run_circuit( # (batch,2**d_actions,)
                # All device and QNode parameters.
                key=subkey,
                device_name=self.device_name,
                shots=self.shots,
                diff_method=self.diff_method,
                # Circuit arguments.
                inputs=x,
                probs_actions=probs_actions,
                weights=self.circ_weights,
                # weights=self.circ_weights[n],
                n_actions=self.n_actions,
                d_actions=self.d_actions,
                d_returns=self.d_returns,
                m_moment=self.m_moment,
                n_layers=self.n_layers,
                U_atoms=self.U_atoms,
            )
            # Filter circuit output based on number of "active" actions.
            # If desired number of actions is not a multiple of 2, then filter out the excess actions on the qubit register (i.e., more actions can be represented on the register and we don't care about them).
            probs_actions = probs_actions[:,jnp.array(self.active_action_idx, dtype=int)] # (batch,n_actions,)
            # Ensure probabilities are normalized after the filter.
            probs_actions = probs_actions / probs_actions.sum(axis=-1, keepdims=True)
            # jax.debug.print("[QuantumNativeQNetworkCascade] Upload {n}: done, {probs_actions}", n=n, probs_actions=probs_actions)
        return probs_actions
    
    @staticmethod
    def circuit(
        inputs: jax.Array, # (d_returns,1,)
        probs_actions: jax.Array,
        weights: jax.Array, # (n_actions,n_layers,d_returns,3,)
        n_actions: int,
        d_actions: int,
        d_returns: int,
        m_moment: int,
        n_layers: int,
        U_atoms: jax.Array,
        ):
        
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
        
        def reset(wires: list[int]):
            """Measure the given qubits to collapse their state, then reset to zero."""
            for w in wires:
                qml.measure(wires=w, reset=True)
        
        def get_ops_for_returns_distribution(
            inputs: jax.Array, # (d_returns,1,)
            weights: jax.Array, # (n_layers,d_returns,3,)
            n_layers: int,
            wires: list[int],
            ):
            """Return-atom distribution unitary.
            
            Layered variational, circular-entanglement, and encoding circuit.
            """
            for l in range(n_layers):
                var(phi=weights[l], wires=wires)
                circ(wires=wires)
                enc(phi=inputs, wires=wires)
            var(phi=weights[-1], wires=wires) # Final variational layer at the end.
        
        def prepare_returns_probability_distribution(
            inputs: jax.Array, # (d_returns,1,)
            weights: jax.Array, # (n_actions,n_layers,d_returns,3,)
            n_actions: int,
            n_layers: int,
            wires_actions: list[int],
            wires_returns: list[int],
            ):
            ops = [
                qml.prod(get_ops_for_returns_distribution)(
                    inputs=inputs,
                    weights=weights[a],
                    n_layers=n_layers,
                    wires=wires_returns,
                )
                for a in range(n_actions)
            ]
            qml.Select(ops, control=wires_actions)
        
        def prepare_m_moment(
            m_moment: int,
            U_atoms: jax.Array,
            wires_returns: list[int],
            ):
            # Apply the (non-unitary) return-atom-value operation.
            # qml.QubitUnitary(U_returns_atoms, wires=wires_returns, unitary_check=False) # We know it's not unitary, so don't check.
            for _ in range(m_moment):
                qml.QubitUnitary(U_atoms, wires=wires_returns, unitary_check=False) # We know it's not unitary, so don't check.
        
        # Create wires.
        num_wires = d_actions + d_returns
        wirelist = list(range(num_wires)) # All wires.
        wiredict = { # Wires separated into blocks for each section.
            'actions': wirelist[:d_actions],
            'returns': wirelist[d_actions:]
        }
        
        # Equal superposition of all actions for each policy.
        # Use `StatePrep` here because there may be fewer "active" actions than the total number of actions supported by the number of qubits in the action register. This ensures that all "inactive" actions have 0 amplitude.
        qml.StatePrep(probs_actions, wires=wiredict['actions'], normalize=True, pad_with=0)
        
        # Prepare returns probability distribution for every policy.
        prepare_returns_probability_distribution(
            inputs=inputs,
            weights=weights,
            n_actions=n_actions,
            n_layers=n_layers,
            wires_actions=wiredict['actions'],
            wires_returns=wiredict['returns'],
        )
        # Prepare the m-th moment of the returns distribution.
        prepare_m_moment(
            m_moment=m_moment,
            U_atoms=U_atoms,
            wires_returns=wiredict['returns'],
        )
        # Measure the return-atom qubits to collapse their state, then reset to zero.
        reset(wires=wiredict['returns'])
        
        return qml.probs(wires=wiredict['actions'])
        # return (
        #     qml.probs(wires=wiredict['actions']),
        #     qml.probs(wires=wiredict['returns']),
        # )

    @staticmethod
    @partial(nnx.jit, static_argnames=[
        # Device and QNode parameters.
        'device_name', 'shots', 'diff_method',
        # Circuit parameters.
        'n_actions',
        'd_actions',
        'd_returns',
        'm_moment',
        'n_layers',
        'U_atoms',
    ])
    def run_circuit(
        # All device and QNode parameters.
        key: jax.Array,
        device_name: str,
        shots: int | None,
        diff_method: str | None,
        # Circuit arguments.
        inputs: jax.Array, # (d_returns,1,)
        weights: jax.Array, # (n_actions,n_layers,d_returns,3,)
        probs_actions: jax.Array,
        n_actions: int,
        d_actions: int,
        d_returns: int,
        m_moment: int,
        n_layers: int,
        U_atoms: jax.Array,
        ):
        device = qml.device(name=device_name, shots=shots, seed=key)
        qnode = qml.QNode(func=QuantumNativeQNetworkCascade.circuit, device=device, interface="jax", diff_method=diff_method)
        qnode_vmap = nnx.vmap(
            f=qnode,
            in_axes=(0, 0, *([None]*7)), # (inputs, probs_actions, weights, n_actions, d_actions, d_returns, m_moment, n_layers, U_atoms)
            out_axes=0, # inputs
        )
        return qnode_vmap(
            inputs=inputs,
            probs_actions=probs_actions,
            weights=weights,
            n_actions=n_actions,
            d_actions=d_actions,
            d_returns=d_returns,
            m_moment=m_moment,
            n_layers=n_layers,
            U_atoms=U_atoms,
        )


# TODO split QnRL into multiple stages: 1) a circuit that uses amplitude kickback to generate an action distribuion using shots>0 and diffmethod=None, 2) a circuit that ingests the action distribuion from #1 and samples the return-atom distribution (which is a weighted average of all the action return-atom distributions) using shots=None and diffmethod='backprop', 3) perform DRL optimization using the return-atom distribution from #2.
@dataclass(kw_only=True)
class QuantumNativeQNetworkSplit(SerializableModule):
    """Split variant of quantum-native (distributional) Q-value network.
    
    Includes the following splits:
    1. Cascaded amplitude kickback circuit for action probability estimation to prevent memory explosion for high values of `n_reuploads` (with `diffmethod=None` and `shots>0`).
    2. Return distribuion generation and weighted return estimation circuit that ingests the action distribuion from #1 (with `diffmethod='backprop'` and `shots=None`).
    """
    din: int
    # n_policies: int
    # """Number of policies to optimize (will require `ceil(log2(n))` qubits)."""
    n_actions: int
    """Number of actions (will require `ceil(log2(n))` qubits)."""
    n_returns: int
    """Number of return atoms (will require `ceil(log2(n))` qubits)."""
    z_min: float
    z_max: float
    m_moment: int
    """m-th moment of the return atoms distribution (`E[Z^m]`)."""
    n_reuploads: int
    """n-th power of the expectation over the return atoms distribution (`E[Z]^n`)."""
    n_layers: int
    """Number of re-upload layers to use for the return-atom distribution unitary."""
    rngs_kwargs: dict[str, int]
    device_name: str = "default.qubit"
    diff_method: str | None = None
    shots: int | None = None
    
    def __post_init__(self):
        self.rngs: nnx.Rngs = nnx.Rngs(**self.rngs_kwargs)
        
        # Compute qubit count based on number of policies, actions, and return atoms.
        # self.d_policies = int(jnp.ceil(jnp.log2(self.n_policies))) # Qubit count for policies to optimize.
        self.d_actions = int(jnp.ceil(jnp.log2(self.n_actions))) # Qubit count for actions.
        self.d_returns = int(jnp.ceil(jnp.log2(self.n_returns))) # Qubit count for return atoms.
        
        # Collect indices of actions based on the total number of actions that can be represented by the number of qubits on the action register.
        self.all_action_idx = list(range(2**self.d_actions)) # All actions represented in the action qubit register.
        self.active_action_idx = list(range(self.n_actions)) # Indices of actions that are being used on the action qubit register.
        self.inactive_action_idx = list(set(self.all_action_idx) - set(self.active_action_idx)) # Indices of actions that are not being used on the action qubit register.
        
        # Collect indices of returns based on the total number of returns that can be represented by the number of qubits on the return register.
        self.all_return_idx = list(range(2**self.d_returns)) # All returns represented in the return qubit register.
        self.active_return_idx = list(range(self.n_returns)) # Indices of returns that are being used on the return qubit register.
        self.inactive_return_idx = list(set(self.all_return_idx) - set(self.active_return_idx)) # Indices of returns that are not being used on the return qubit register.
        
        # Create list of return atoms.
        self.atoms = jnp.linspace(start=self.z_min, stop=self.z_max, num=self.n_returns)
        # Create non-linear trace non-increasing Kraus matrix of returns atoms.
        atoms_values = nnx.softmax(self.atoms) # Normalize to probability distribution.
        atoms_values = jnp.sqrt(atoms_values * self.n_returns) # Ensure that L2-norm of quantum state amplitude will produce the atom probability (i.e., $|amp|^2 = prob$).
        atoms_values = atoms_values / atoms_values.max() # Ensure maximum value is 1.
        if self.n_returns != 2**self.d_returns: # Pad with zeros if return atoms shape is not a power of 2.
            atoms_values = jnp.pad(atoms_values, (0,2**self.d_returns - self.n_returns), mode='constant')
        self.U_atoms = jnp.diag(atoms_values)
        
        # TODO returns probability matrix with shape (n_policy,n_action,n_return).
        # self.returns_probs_matrix = nnx.Param(
        #     nnx.softmax( # Normalize to make it a probability vector (sums to 1).
        #         x=jax.random.normal( # Atom probabilities sampled from normal distribution.
        #             key=self.rngs.params(),
        #             # shape=(self.n_policies, self.n_actions, self.n_returns),
        #             shape=(self.n_actions, self.n_returns),
        #         ),
        #         axis=-1, # Softmax over `n_returns` dimension.
        #     )
        # )
        
        # Weights for return-atom distribution generation.
        # Every action gets their own return-atom distribution unitary.
        self.circ_weights_shape = (self.n_actions,self.n_layers,self.d_returns,3,) # Each qubit has (Rx, Ry, Rz).
        self.circ_weights = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_shape, minval=0, maxval=jnp.pi))
        ####
        # self.circ_weights_base = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_shape, minval=0, maxval=jnp.pi))
        # self.circ_weights = [
        #     self.circ_weights_base for _ in range(self.n_reuploads)
        # ] # Same parameter for each reupload layer.
        # # self.circ_weights = [
        # #     nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_shape, minval=0, maxval=jnp.pi))
        # #     for _ in range(self.n_reuploads)
        # # ] # (n_reuploads,n_actions,n_layers,d_returns,3,)
        
        
        # Inputs to circuit will be encoded onto the return-atom wires.
        self.circ_inputs_shape = (self.d_returns,1,) # Each qubit has (Rx,).
        self.layer_project_input = nnx.Linear(in_features=self.din, out_features=reduce(lambda x,y: x*y, self.circ_inputs_shape), rngs=self.rngs)
    
    @nnx.jit
    def __call__(self, x): # (batch,din,)
        return self.call_gen_action_dist(x) # Generation action distribution.
        # # Project input into shape that is compatible with the quantum circuit.
        # x = self.layer_project_input(x)
        # # Input must be batched for input into the circuit.
        # x = jnp.reshape(x, (-1, *self.circ_inputs_shape,)) # (batch,d_returns,1,)
        # # Prepare the n-th power of the m-th moment of the returns distribution.
        # key = self.rngs.default() # Generate new random key for circuit measurement.
        # probs_actions = jnp.ones(shape=(x.shape[0], self.n_actions,), dtype='float64')
        # for _ in range(self.n_reuploads):
        #     # jax.debug.print("[QuantumNativeQNetworkCascade] Upload {n}: start, {probs_actions}", n=n, probs_actions=probs_actions)
        #     # Run quantum circuit.
        #     key, subkey = jax.random.split(key=key)
        #     probs_actions = self.run_circuit( # (batch,2**d_actions,)
        #         # All device and QNode parameters.
        #         key=subkey,
        #         device_name=self.device_name,
        #         shots=self.shots,
        #         diff_method=self.diff_method,
        #         # Circuit arguments.
        #         inputs=x,
        #         probs_actions=probs_actions,
        #         weights=self.circ_weights,
        #         # weights=self.circ_weights[n],
        #         n_actions=self.n_actions,
        #         d_actions=self.d_actions,
        #         d_returns=self.d_returns,
        #         m_moment=self.m_moment,
        #         n_layers=self.n_layers,
        #         U_atoms=self.U_atoms,
        #     )
        #     # Filter circuit output based on number of "active" actions.
        #     # If desired number of actions is not a multiple of 2, then filter out the excess actions on the qubit register (i.e., more actions can be represented on the register and we don't care about them).
        #     probs_actions = probs_actions[:,jnp.array(self.active_action_idx, dtype=int)] # (batch,n_actions,)
        #     # Ensure probabilities are normalized after the filter.
        #     probs_actions = probs_actions / probs_actions.sum(axis=-1, keepdims=True)
        #     # jax.debug.print("[QuantumNativeQNetworkCascade] Upload {n}: done, {probs_actions}", n=n, probs_actions=probs_actions)
        # return probs_actions
    
    
    @nnx.jit
    def call_gen_action_dist(self, x): # (batch,din,)
        """Generate action distribuion from input observation."""
        # Project input into shape that is compatible with the quantum circuit.
        x = self.layer_project_input(x)
        # Input must be batched for input into the circuit.
        x = jnp.reshape(x, (-1, *self.circ_inputs_shape,)) # (batch,d_returns,1,)
        # Prepare the n-th power of the m-th moment of the returns distribution.
        key = self.rngs.default() # Generate new random key for circuit measurement.
        probs_actions = jnp.ones(shape=(x.shape[0], self.n_actions,), dtype='float64')
        for _ in range(self.n_reuploads):
            # jax.debug.print("[QuantumNativeQNetworkCascade] Upload {n}: start, {probs_actions}", n=n, probs_actions=probs_actions)
            # Run quantum circuit.
            key, subkey = jax.random.split(key=key)
            probs_actions = self.run_circuit_action_dist( # (batch,2**d_actions,)
                # All device and QNode parameters.
                key=subkey,
                device_name=self.device_name,
                # shots=self.shots,
                # diff_method=self.diff_method,
                shots=100, diff_method=None, # Shots with no diffmethod ensures that memory doesn't explore with reset operations.
                # Circuit arguments.
                inputs=x,
                probs_actions=probs_actions,
                weights=self.circ_weights,
                n_actions=self.n_actions,
                d_actions=self.d_actions,
                d_returns=self.d_returns,
                m_moment=self.m_moment,
                n_layers=self.n_layers,
                U_atoms=self.U_atoms,
            )
            # Filter circuit output based on number of "active" actions.
            # If desired number of actions is not a multiple of 2, then filter out the excess actions on the qubit register (i.e., more actions can be represented on the register and we don't care about them).
            probs_actions = probs_actions[:,jnp.array(self.active_action_idx, dtype=int)] # (batch,n_actions,)
            # Ensure probabilities are normalized after the filter.
            probs_actions = probs_actions / probs_actions.sum(axis=-1, keepdims=True)
            # jax.debug.print("[QuantumNativeQNetworkCascade] Upload {n}: done, {probs_actions}", n=n, probs_actions=probs_actions)
        return probs_actions
    
    @nnx.jit
    def call_gen_return_dist(self, x): # (batch,din,)
        """Generate return distribuion from input observation."""
        
        # TODO freeze weights for action distribution generation?
        # probs_actions = self.call_gen_action_dist(x)
        probs_actions = self.call_gen_action_dist(jax.lax.stop_gradient(x))
        # # Stop gradient.
        # probs_actions = jax.lax.stop_gradient(probs_actions)
        
        # Project input into shape that is compatible with the quantum circuit.
        x = self.layer_project_input(x)
        
        # Input must be batched for input into the circuit.
        x = jnp.reshape(x, (-1, *self.circ_inputs_shape,)) # (batch,d_returns,1,)
        
        key = self.rngs.default() # Generate new random key for circuit measurement.
        key, subkey = jax.random.split(key=key)
        probs_returns = self.run_circuit_return_dist( # (batch,2**d_returns,)
            # All device and QNode parameters.
            key=subkey,
            device_name=self.device_name,
            # shots=self.shots, diff_method=self.diff_method,
            shots=None, diff_method='backprop',
            # Circuit arguments.
            inputs=x,
            probs_actions=probs_actions,
            weights=self.circ_weights,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            n_layers=self.n_layers,
        )
        # Filter circuit output based on number of "active" returns.
        # If desired number of returns is not a multiple of 2, then filter out the excess returns on the qubit register (i.e., more returns can be represented on the register and we don't care about them).
        probs_returns = probs_returns[:,jnp.array(self.active_return_idx, dtype=int)] # (batch,n_returns,)
        # Ensure probabilities are normalized after the filter.
        probs_returns = probs_returns / probs_returns.sum(axis=-1, keepdims=True)
        return probs_returns
    
    @nnx.jit
    def call_gen_return_dist_NEW(self, x, probs_actions): # (batch,din,)
        """Generate return distribuion from input observation."""
        
        # Project input into shape that is compatible with the quantum circuit.
        x = self.layer_project_input(x)
        
        # Input must be batched for input into the circuit.
        x = jnp.reshape(x, (-1, *self.circ_inputs_shape,)) # (batch,d_returns,1,)
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()

        key, subkey = jax.random.split(key=key)
        probs_returns = self.run_circuit_return_dist( # (batch,2**d_returns,)
            # All device and QNode parameters.
            key=subkey,
            device_name=self.device_name,
            # shots=self.shots, diff_method=self.diff_method,
            shots=None, diff_method='backprop',
            # Circuit arguments.
            inputs=x,
            probs_actions=probs_actions,
            weights=self.circ_weights,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            n_layers=self.n_layers,
        )
        # Filter circuit output based on number of "active" returns.
        # If desired number of returns is not a multiple of 2, then filter out the excess returns on the qubit register (i.e., more returns can be represented on the register and we don't care about them).
        probs_returns = probs_returns[:,jnp.array(self.active_return_idx, dtype=int)] # (batch,n_returns,)
        # Ensure probabilities are normalized after the filter.
        probs_returns = probs_returns / probs_returns.sum(axis=-1, keepdims=True)
        return probs_returns
    
    
    @staticmethod
    def circuit_layer_var(phi: jax.Array, wires: list[int]):
        """Variational layer.

        Args:
            phi (jax.Array): Variational parameters for the layer of shape `(num_wires,3)`.
            wires (list[int]): List of wires.
        """
        for w in wires:
            qml.RX(phi=phi[w,0], wires=w)
            qml.RY(phi=phi[w,1], wires=w)
            qml.RZ(phi=phi[w,2], wires=w)
    
    @staticmethod
    def circuit_layer_circ(wires: list[int]):
        """Circular entanglement layer using `CZ`.

        Args:
            wires (list[int]): List of wires.
        """
        for w0, w1 in zip(wires, wires[1:]):
            qml.CZ(wires=[w0, w1])
        if len(wires) != 2:
            qml.CZ(wires=[wires[0], wires[-1]])
    
    @staticmethod
    def circuit_layer_enc(phi: jax.Array, wires: list[int]):
        """Encoding layer using `RX`.

        Args:
            phi (jax.Array): Encoding parameters for the layer of shape `(num_wires,1)`.
            wires (list[int]): _description_
        """
        for w in wires:
            qml.RX(phi=phi[w,0], wires=w)
    
    @staticmethod
    def circuit_layer_reset(wires: list[int]):
        """Measure the given qubits to collapse their state, then reset to zero."""
        for w in wires:
            qml.measure(wires=w, reset=True)
    
    @staticmethod
    def get_ops_for_returns_distribution(
        inputs: jax.Array, # (d_returns,1,)
        weights: jax.Array, # (n_layers,d_returns,3,)
        n_layers: int,
        wires: list[int],
        ):
        """Return-atom distribution unitary.
        
        Generates a return distribuion encoded on the given wires.
        
        The schema of this operation is:
            - `n_layers+1` Variational layers
            - `n_layers` Circular-entanglement layers
            - `n_layers` Input encoding layers
        """
        for l in range(n_layers):
            QuantumNativeQNetworkSplit.circuit_layer_var(phi=weights[l], wires=wires)
            QuantumNativeQNetworkSplit.circuit_layer_circ(wires=wires)
            QuantumNativeQNetworkSplit.circuit_layer_enc(phi=inputs, wires=wires)
        QuantumNativeQNetworkSplit.circuit_layer_var(phi=weights[-1], wires=wires) # Final variational layer at the end.
    
    @staticmethod
    def prepare_m_moment(
        m_moment: int,
        U_atoms: jax.Array,
        wires_returns: list[int],
        ):
        """Prepare m-th moment of the return distribuion encoded onto the given wires."""
        # Apply the (non-unitary) return-atom-value operation.
        # qml.QubitUnitary(U_returns_atoms, wires=wires_returns, unitary_check=False) # We know it's not unitary, so don't check.
        for _ in range(m_moment):
            qml.QubitUnitary(U_atoms, wires=wires_returns, unitary_check=False) # We know it's not unitary, so don't check.
    
    @staticmethod
    def prepare_returns_probability_distribution(
        inputs: jax.Array, # (d_returns,1,)
        weights: jax.Array, # (n_actions,n_layers,d_returns,3,)
        n_actions: int,
        n_layers: int,
        wires_actions: list[int],
        wires_returns: list[int],
        ):
        """Prepare return probability distribuion for every action."""
        ops = [
            qml.prod(QuantumNativeQNetworkSplit.get_ops_for_returns_distribution)(
                inputs=inputs,
                weights=weights[a],
                n_layers=n_layers,
                wires=wires_returns,
            )
            for a in range(n_actions)
        ]
        qml.Select(ops, control=wires_actions)
    
    @staticmethod
    def circuit_action_dist(
        inputs: jax.Array, # (d_returns,1,)
        probs_actions: jax.Array,
        weights: jax.Array, # (n_actions,n_layers,d_returns,3,)
        n_actions: int,
        d_actions: int,
        d_returns: int,
        m_moment: int,
        n_layers: int,
        U_atoms: jax.Array,
        ):
        """Quantum circuit for estimating the action distribution via m-th moment of return distribuion followed by amplitude kickback trick with respect to an input observation."""
        
        # Create wires.
        num_wires = d_actions + d_returns
        wirelist = list(range(num_wires)) # All wires.
        wiredict = { # Wires separated into blocks for each section.
            'actions': wirelist[:d_actions],
            'returns': wirelist[d_actions:]
        }
        
        # Weighted superposition of all actions.
        # Use `StatePrep` here because there may be fewer "active" actions than the total number of actions supported by the number of qubits in the action register. This ensures that all "inactive" actions have 0 amplitude.
        qml.StatePrep(probs_actions, wires=wiredict['actions'], normalize=True, pad_with=0)
        
        # Prepare returns probability distribution for every policy.
        QuantumNativeQNetworkSplit.prepare_returns_probability_distribution(
            inputs=inputs,
            weights=weights,
            n_actions=n_actions,
            n_layers=n_layers,
            wires_actions=wiredict['actions'],
            wires_returns=wiredict['returns'],
        )
        # Prepare the m-th moment of the returns distribution.
        QuantumNativeQNetworkSplit.prepare_m_moment(
            m_moment=m_moment,
            U_atoms=U_atoms,
            wires_returns=wiredict['returns'],
        )
        # Measure the return-atom qubits to collapse their state, then reset to zero.
        QuantumNativeQNetworkSplit.circuit_layer_reset(wires=wiredict['returns'])
        
        # Generate action distribuion via measurements across the action wires.
        return qml.probs(wires=wiredict['actions'])
    
    @staticmethod
    def circuit_return_dist(
        inputs: jax.Array, # (d_returns,1,)
        probs_actions: jax.Array,
        weights: jax.Array, # (n_actions,n_layers,d_returns,3,)
        n_actions: int,
        d_actions: int,
        d_returns: int,
        n_layers: int,
        ):
        """Quantum circuit for generating the return distribution that is weighted via the probability amplitudes of the action space with respect to an input observation."""
        
        # Create wires.
        num_wires = d_actions + d_returns
        wirelist = list(range(num_wires)) # All wires.
        wiredict = { # Wires separated into blocks for each section.
            'actions': wirelist[:d_actions],
            'returns': wirelist[d_actions:]
        }
        
        # Weighted superposition of all actions.
        # Use `StatePrep` here because there may be fewer "active" actions than the total number of actions supported by the number of qubits in the action register. This ensures that all "inactive" actions have 0 amplitude.
        qml.StatePrep(probs_actions, wires=wiredict['actions'], normalize=True, pad_with=0)
        
        # Prepare returns probability distribution for every policy.
        QuantumNativeQNetworkSplit.prepare_returns_probability_distribution(
            inputs=inputs,
            weights=weights,
            n_actions=n_actions,
            n_layers=n_layers,
            wires_actions=wiredict['actions'],
            wires_returns=wiredict['returns'],
        )
        
        # Generate weighted return distribuion via measurements across the return wires.
        return qml.probs(wires=wiredict['returns'])

    @staticmethod
    @partial(nnx.jit, static_argnames=[
        # Device and QNode parameters.
        'device_name', 'shots', 'diff_method',
        # Circuit parameters.
        'n_actions',
        'd_actions',
        'd_returns',
        'm_moment',
        'n_layers',
        'U_atoms',
    ])
    def run_circuit_action_dist(
        # All device and QNode parameters.
        key: jax.Array,
        device_name: str,
        shots: int | None,
        diff_method: str | None,
        # Circuit arguments.
        inputs: jax.Array, # (d_returns,1,)
        weights: jax.Array, # (n_actions,n_layers,d_returns,3,)
        probs_actions: jax.Array,
        n_actions: int,
        d_actions: int,
        d_returns: int,
        m_moment: int,
        n_layers: int,
        U_atoms: jax.Array,
        ):
        device = qml.device(name=device_name, shots=shots, seed=key)
        qnode = qml.QNode(func=QuantumNativeQNetworkSplit.circuit_action_dist, device=device, interface="jax", diff_method=diff_method)
        qnode_vmap = nnx.vmap(
            f=qnode,
            in_axes=(0, 0, *([None]*7)), # (inputs, probs_actions, weights, n_actions, d_actions, d_returns, m_moment, n_layers, U_atoms)
            out_axes=0, # inputs
        )
        return qnode_vmap(
            inputs=inputs,
            probs_actions=probs_actions,
            weights=weights,
            n_actions=n_actions,
            d_actions=d_actions,
            d_returns=d_returns,
            m_moment=m_moment,
            n_layers=n_layers,
            U_atoms=U_atoms,
        )
    
    @staticmethod
    @partial(nnx.jit, static_argnames=[
        # Device and QNode parameters.
        'device_name', 'shots', 'diff_method',
        # Circuit parameters.
        'n_actions',
        'd_actions',
        'd_returns',
        'n_layers',
    ])
    def run_circuit_return_dist(
        # All device and QNode parameters.
        key: jax.Array,
        device_name: str,
        shots: int | None,
        diff_method: str | None,
        # Circuit arguments.
        inputs: jax.Array, # (d_returns,1,)
        weights: jax.Array, # (n_actions,n_layers,d_returns,3,)
        probs_actions: jax.Array,
        n_actions: int,
        d_actions: int,
        d_returns: int,
        n_layers: int,
        ):
        device = qml.device(name=device_name, shots=shots, seed=key)
        qnode = qml.QNode(func=QuantumNativeQNetworkSplit.circuit_return_dist, device=device, interface="jax", diff_method=diff_method)
        qnode_vmap = nnx.vmap(
            f=qnode,
            in_axes=(0, 0, *([None]*5)), # (inputs, probs_actions, weights, n_actions, d_actions, d_returns, n_layers)
            out_axes=0, # inputs
        )
        return qnode_vmap(
            inputs=inputs,
            probs_actions=probs_actions,
            weights=weights,
            n_actions=n_actions,
            d_actions=d_actions,
            d_returns=d_returns,
            n_layers=n_layers,
        )