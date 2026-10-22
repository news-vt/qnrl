from dataclasses import dataclass, field
from functools import partial, reduce
from typing import Callable, Any

import jax
from flax import nnx
import jax.numpy as jnp
import pennylane as qp

from .serializable import SerializableModule


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
        for i, w in enumerate(wires):
            qp.RZ(phi=phi[i,0], wires=w)
            qp.RY(phi=phi[i,1], wires=w)
            qp.RZ(phi=phi[i,2], wires=w)
    
    def circ(wires: list[int]):
        """Circular entanglement layer using `CZ`.

        Args:
            wires (list[int]): List of wires.
        """
        for w0, w1 in zip(wires, wires[1:]):
            qp.CZ(wires=[w0, w1]) # Control w0, target w1.
        if len(wires) != 2:
            qp.CZ(wires=[wires[-1], wires[0]]) # Control w[-1], target w[0].
    
    def enc(phi: jax.Array, wires: list[int]):
        """Encoding layer using `RX`.

        Args:
            phi (jax.Array): Encoding parameters for the layer of shape `(num_wires,1)`.
            wires (list[int]): _description_
        """
        for i, w in enumerate(wires):
            qp.RX(phi=phi[i,0], wires=w)
    
    # Create list of wire indices.
    wires = list(range(num_wires))
    
    # Cascaded variational, circular entanglement, and encoding layers.
    for l in range(num_layers):
        var(phi=weights[l], wires=wires)
        circ(wires=wires)
        enc(phi=inputs[l], wires=wires)
    # Final variational layer at the end.
    var(phi=weights[-1], wires=wires)
    
    return [qp.expval(qp.PauliZ(wires=w)) for w in wires]


@partial(nnx.jit, static_argnames=[
    # Device and QNode parameters.
    'device_name', 'shots', 'diff_method',
    # Circuit parameters.
    'num_wires',
    'num_layers',
])
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
    device = qp.device(name=device_name, shots=shots, seed=key)
    qnode = qp.QNode(func=circuit, device=device, interface="jax", diff_method=diff_method)
    return qnode(
        inputs=inputs,
        weights=weights,
        num_wires=num_wires,
        num_layers=num_layers,
    )


@partial(nnx.jit, static_argnames=[
    # Device and QNode parameters.
    'device_name', 'shots', 'diff_method',
    # Circuit parameters.
    'num_wires',
    'num_layers',
])
def batched_run_circuit(
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
    qnode_vmap = nnx.vmap(
        f=run_circuit,
        in_axes=(
            0, # key
            None, # device_name
            None, # shots
            None, # diff_method
            0, # inputs
            None, # weights
            None, # num_wires
            None, # num_layers
            ),
        out_axes=0, # inputs
    )
    return qnode_vmap(
        key,
        device_name,
        shots,
        diff_method,
        inputs=inputs,
        weights=weights,
        num_wires=num_wires,
        num_layers=num_layers,
    )


@dataclass(kw_only=True)
class QuantumDistQNetworkLinear(SerializableModule):
    """Quantum Distributional Q-learning model using linear (i.e., fully connected) input and output projection layers."""
    din: int
    num_actions: int
    num_atoms: int
    num_wires: int
    num_layers: int
    input_preprocess: SerializableModule | None = None
    """Nested module for input preprocessing (can be trainable)."""
    rngs_kwargs: dict[str, int]
    device_name: str = "default.qubit"
    diff_method: str | None = "backprop"
    shots: int | None = None

    def __post_init__(self):
        self.circ_observation_shape_in = self.din # Dimension of the input data.
        self.circ_observation_shape_out = (self.num_layers,self.num_wires,1,) # Each qubit has (Rx,).
        self.circ_weights_shape = (self.num_layers+1,self.num_wires,3,) # Each qubit has (Rz, Ry, Rz).
        self.circ_outputs_shape_in = self.num_wires
        self.circ_outputs_shape_out = (self.num_actions, self.num_atoms,)
        
        # Create parameters that depend on random seed.
        self.reseed(self.rngs_kwargs)
    
    def pre_reseed(self, rngs_kwargs: dict[str, int]):
        pass

    def post_reseed(self, rngs_kwargs: dict[str, int]):
        self.circ_weights = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_shape, minval=0, maxval=jnp.pi))
        
        self.layer_project_input = nnx.Linear(in_features=self.circ_observation_shape_in, out_features=reduce(lambda x,y: x*y, self.circ_observation_shape_out), rngs=self.rngs)
        self.layer_project_output = nnx.Linear(in_features=self.circ_outputs_shape_in, out_features=reduce(lambda x,y: x*y, self.circ_outputs_shape_out), rngs=self.rngs)

    def reseed(self, rngs_kwargs: dict[str, int]):
        """Re-create model parameters using a dictionary of `nnx.Rngs` keyword arguments."""
        self.pre_reseed(rngs_kwargs=rngs_kwargs)
        self.rngs_kwargs = rngs_kwargs
        self.rngs: nnx.Rngs = nnx.Rngs(**self.rngs_kwargs)
        self.post_reseed(rngs_kwargs=rngs_kwargs)
    
    @nnx.jit
    def __call__(self, x):
        
        # Preprocess input data.
        if self.input_preprocess is not None:
            x = self.input_preprocess(x)
        
        x = self.layer_project_input(x)
        # Input must be batched for input into the circuit.
        x = jnp.reshape(x, (-1, *self.circ_observation_shape_out,))
        
        # Track the batch size.
        batch_size = x.shape[0]
        
        # Circuit input activation.
        x = jnp.tanh(x)
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()

        # Create a batch of keys.
        subkeys = jax.random.split(key=key, num=batch_size)
        
        # Output of circuit will be reverse-shaped list with shape `(num_wires,batch,)` due to how batching is handled with pennylane measurements.
        x = batched_run_circuit(
            key=subkeys,
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
        x = self.layer_project_output(x) # (batch,n_actions*n_atoms,)
        x = x.reshape((-1, *self.circ_outputs_shape_out)) # (batch, n_actions, n_atoms,)
        x = nnx.softmax(x, axis=-1) # Distribution over the atoms for every action.
        return x