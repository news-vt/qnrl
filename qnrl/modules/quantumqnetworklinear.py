from dataclasses import dataclass, field
from functools import partial, reduce
from typing import Callable, Any

import jax
from flax import nnx
import jax.numpy as jnp
import pennylane as qml

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
            qml.RZ(phi=phi[i,0], wires=w)
            qml.RY(phi=phi[i,1], wires=w)
            qml.RZ(phi=phi[i,2], wires=w)
    
    def circ(wires: list[int]):
        """Circular entanglement layer using `CZ`.

        Args:
            wires (list[int]): List of wires.
        """
        for w0, w1 in zip(wires, wires[1:]):
            qml.CZ(wires=[w0, w1]) # Control w0, target w1.
        if len(wires) != 2:
            qml.CZ(wires=[wires[-1], wires[0]]) # Control w[-1], target w[0].
    
    def enc(phi: jax.Array, wires: list[int]):
        """Encoding layer using `RX`.

        Args:
            phi (jax.Array): Encoding parameters for the layer of shape `(num_wires,1)`.
            wires (list[int]): _description_
        """
        for i, w in enumerate(wires):
            qml.RX(phi=phi[i,0], wires=w)
    
    # Create list of wire indices.
    wires = list(range(num_wires))
    
    # Cascaded variational, circular entanglement, and encoding layers.
    for l in range(num_layers):
        var(phi=weights[l], wires=wires)
        circ(wires=wires)
        enc(phi=inputs[l], wires=wires)
    # Final variational layer at the end.
    var(phi=weights[-1], wires=wires)
    
    return [qml.expval(qml.PauliZ(wires=w)) for w in wires]


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
    device = qml.device(name=device_name, shots=shots, seed=key)
    qnode = qml.QNode(func=circuit, device=device, interface="jax", diff_method=diff_method)
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

# https://discuss.pennylane.ai/t/trouble-calling-qjit-compiled-qnode-inside-a-nnx-jit-compiled-flax-nnx-module/7853
# https://discuss.pennylane.ai/t/error-combining-jax-with-multiple-devices/4934
# https://discuss.pennylane.ai/t/facing-issues-with-jax-jitting-the-optimization-loop/4274
@dataclass(kw_only=True)
class QuantumQNetworkLinear(SerializableModule):
    """Quantum Q-learning model using linear (i.e., fully connected) input and output projection layers."""
    din: int
    num_actions: int
    num_wires: int
    num_layers: int
    rngs_kwargs: dict[str, int]
    device_name: str = "default.qubit"
    diff_method: str | None = "backprop"
    shots: int | None = None

    def __init__(self,
        din: int,
        num_actions: int,
        num_wires: int,
        num_layers: int,
        *,
        rngs_kwargs: dict[str, int],
        device_name: str = "default.qubit",
        diff_method: str | None = "backprop",
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
        
        self.circ_inputs_shape = (num_layers,num_wires,1,) # Each qubit has (Rx,).
        self.circ_weights_shape = (num_layers+1,num_wires,3,) # Each qubit has (Rz, Ry, Rz).
        self.circ_outputs_shape = num_wires
        
        self.layer_project_input = nnx.Linear(in_features=din, out_features=reduce(lambda x,y: x*y, self.circ_inputs_shape), rngs=self.rngs)
        self.layer_project_output = nnx.Linear(in_features=num_wires, out_features=num_actions, rngs=self.rngs)
        
        self.circ_weights = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_shape, minval=0, maxval=jnp.pi))
    
    @nnx.jit
    def __call__(self, x):
        # Track the batch size.
        batch_size = x.shape[0] if len(x.shape) > 1 else 1 if len(x.shape) > 1 else 1
        
        x = self.layer_project_input(x)
        # Input must be batched for input into the circuit.
        x = jnp.reshape(x, (-1, *self.circ_inputs_shape,))
        
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
        x = self.layer_project_output(x)
        return x