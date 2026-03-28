from dataclasses import dataclass
from functools import partial, reduce
from itertools import combinations
from typing import Literal

import pennylane as qml
import numpy as np
import jax
from jax import numpy as jnp # JAX version of NumPy.
from flax import nnx # Neural networks in JAX.

from .serializable import SerializableModule, activation_function_dispatcher



class PhaseShiftX(qml.operation.Operation):
    """Variation of `qml.PhaseShift` with Pauli X rotation.
    
    This is useful because the same rotation angle `phi` is used for both the Pauli rotation and the global phase operation.
    """
    num_wires = 1
    num_params = 1
    
    @staticmethod
    def compute_decomposition(phi: qml.typing.TensorLike, wires: qml.wires.WiresLike) -> qml.operation.Operator:
        return [qml.RX(phi, wires=wires), qml.GlobalPhase(-phi / 2.)]


class PhaseShiftY(qml.operation.Operation):
    """Variation of `qml.PhaseShift` with Pauli Y rotation.
    
    This is useful because the same rotation angle `phi` is used for both the Pauli rotation and the global phase operation.
    """
    num_wires = 1
    num_params = 1
    
    @staticmethod
    def compute_decomposition(phi: qml.typing.TensorLike, wires: qml.wires.WiresLike) -> qml.operation.Operator:
        return [qml.RY(phi, wires=wires), qml.GlobalPhase(-phi / 2.)]


OP_STRINGS_ONE_QUBIT: dict[str, type[qml.operation.Operation]] = {
    'rx': qml.RX,
    'ry': qml.RY,
    'rz': qml.RZ,
    'globalphase': qml.GlobalPhase,
    'phaseshift': qml.PhaseShift,
    'pz': qml.PhaseShift, # Alias for 'phaseshift'.
    'px': PhaseShiftX,
    'py': PhaseShiftY,
    'rot': qml.Rot,
}
"""Link PennyLane one-qubit (parameterized) operations to string names."""


type OpTypeOneQubit = Literal[*tuple(OP_STRINGS_ONE_QUBIT.keys())]


OP_STRINGS_TWO_QUBIT: dict[str, type[qml.ops.op_math.ControlledOp]] = {
    'cnot': qml.CNOT,
    'cx': qml.CNOT,
    'cy': qml.CY,
    'cz': qml.CZ,
}
"""Link PennyLane two-qubit operations to string names."""


type OpTypeTwoQubit = Literal[*tuple(OP_STRINGS_TWO_QUBIT.keys())]


def get_op_from_string_one_qubit(op: OpTypeOneQubit) -> type[qml.operation.Operation]:
    """Map a one-qubit (parameterized) operation from string name to operation class type."""
    try:
        op_fn = OP_STRINGS_ONE_QUBIT[op.lower()]
        return op_fn
    except:
        raise ValueError(f"Unsupported one-qubit parameterized operation string: {op}")


def get_op_from_string_two_qubit(op: OpTypeTwoQubit) -> type[qml.ops.op_math.ControlledOp]:
    """Map a two-qubit operation from string name to operation class type."""
    try:
        op_fn = OP_STRINGS_TWO_QUBIT[op.lower()]
        return op_fn
    except:
        raise ValueError(f"Unsupported two-qubit operation string: {op}")


def get_op_from_string(op: OpTypeOneQubit) -> type[qml.operation.Operation]:
    """Map a parameterized operation from string name to operation class type."""
    try:
        op_fn = OP_STRINGS_ONE_QUBIT[op.lower()]
        return op_fn
    except:
        raise ValueError(f"Unsupported parameterized operation string: {op}")


def get_op_num_params(op: str) -> int:
    obj = get_op_from_string_one_qubit(op)
    return obj.num_params


def get_ops_num_params(ops: tuple[str,...]) -> int:
    """Get total number of parameters for operations in tuple."""
    return sum(get_op_from_string_one_qubit(op_string).num_params for op_string in ops)


def get_ops_params_shape(wires: list[int], ops: tuple[str,...]):
    num_params = get_ops_num_params(ops)
    num_wires = len(wires)
    return (num_wires, num_params,)


def circuit_layer_ent_neighbor(wires: list[int], op: OpTypeTwoQubit, reverse: bool = False):
    """Next-neighbor entanglement layer.

    Args:
        wires (list[int]): List of wires.
    """
    op_func = get_op_from_string_two_qubit(op=op)
    
    gen = zip(wires, wires[1:])
    if reverse:
        gen = list(reversed(list(gen)))
    for w0, w1 in gen:
        op_func(wires=[w0, w1]) # Control w0, target w1.


def circuit_layer_ent_circ(wires: list[int], op: OpTypeTwoQubit):
    """Circular entanglement layer.

    Args:
        wires (list[int]): List of wires.
    """
    op_func = get_op_from_string_two_qubit(op=op)
    
    for w0, w1 in zip(wires, wires[1:]):
        op_func(wires=[w0, w1]) # Control w0, target w1.
    if len(wires) != 2:
        op_func(wires=[wires[-1], wires[0]]) # Control w[-1], target w[0].


def circuit_layer_ent_circ_offset(wires: list[int], imprimitive_offset: int, op: OpTypeTwoQubit):
    """Circular entanglement that is governed by an offset index, defining which qubits are entangled (with circular rollover).

    Args:
        wires (list[int]): List of wires.
    """
    op_func = get_op_from_string_two_qubit(op=op)
    
    wires = qml.wires.Wires(wires)
    for i in range(len(wires)):
        act_on = wires.subset([i, i + imprimitive_offset], periodic_boundary=True)
        op_func(wires=act_on)


def circuit_layer_op(phi: jax.Array, wires: list[int], pauli: tuple[str,...]):
    """Operational layer `U |psi>`.

    Args:
        phi (jax.Array): Variational parameters for the layer of shape `(num_wires,len(pauli),)`.
        wires (list[int]): List of wires.
    """
    param_idx = 0
    for op_str in pauli:
        op_num_params = get_op_num_params(op_str)
        op_fn = get_op_from_string(op_str)
        idx_offset = param_idx + op_num_params
        idx_slice = slice(param_idx, idx_offset)
        for i, w in enumerate(wires):
            op_fn(*phi[i,idx_slice], wires=w)
        param_idx = idx_offset


def circuit_layer_reset(wires: list[int]):
    """Measure the given qubits to collapse their state, then reset to zero."""
    for w in wires:
        qml.measure(wires=w, reset=True)


def get_ops_for_returns_distribution(
    inputs: jax.Array, # (n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_layers, d_returns, num_var_features,)
    n_layers: int,
    wires: list[int],
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    ):
    """Return-atom distribution unitary.
    
    Generates a return distribuion encoded on the given wires.
    
    The schema of this operation is:
        - Encoding layer
        - Entanglement layer
        - Variational layer
    """
    
    imprimitive_offset_by_layer = tuple((l % (len(wires) - 1)) + 1 for l in range(n_layers))
    
    imprimitive_type = None
    if imprimitive is not None and "-" in imprimitive:
        imprimitive_type, imprimitive = imprimitive.split("-")
    
    for l in range(n_layers):
        
        # Encoder as a function of the layer number.
        circuit_layer_op(phi=inputs[l], wires=wires, pauli=pauli_enc)
        
        # Entangle.
        if imprimitive is not None:
            
            match imprimitive_type:
                case "circ" | None: # Defaults to circular entanglement.
                    circuit_layer_ent_circ(wires=wires, op=imprimitive)
                case "offset":
                    circuit_layer_ent_circ_offset(wires=wires, op=imprimitive, imprimitive_offset=imprimitive_offset_by_layer[l])
                case _:
                    raise ValueError(f"Unsupported imprimitive type: {imprimitive_type}")
        
        # Variation.
        circuit_layer_op(phi=weights[l], wires=wires, pauli=pauli_var)


def prepare_m_moment(
    m_moment: int,
    U_atoms: jax.Array,
    wires_returns: list[int],
    ):
    """Prepare m-th moment of the return distribuion encoded onto the given wires."""
    # Apply the (non-unitary) return-atom-value operation.
    for _ in range(m_moment):
        qml.QubitUnitary(U_atoms, wires=wires_returns, unitary_check=False) # We know it's not unitary, so don't check.

def prepare_returns_probability_distribution(
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    n_actions: int,
    n_layers: int,
    wires_actions: list[int],
    wires_returns: list[int],
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
    ):
    """Prepare return probability distribuion for every action."""
    ###
    # Generate unique return-atom distribution for every action.
    ###
    ops = [
        qml.prod(get_ops_for_returns_distribution)(
            inputs=inputs[a],
            weights=weights[a],
            n_layers=n_layers,
            wires=wires_returns,
            pauli_enc=pauli_enc,
            pauli_var=pauli_var,
            imprimitive=imprimitive,
        )
        for a in range(n_actions)
    ]
    qml.Select(ops, control=wires_actions)
    
    ###
    # Apply operations on all action-return distributions.
    ###
    # Quantum Fourier Transform.
    if use_qft:
        qml.QFT(wires=wires_returns)

def circuit_action_dist(
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    probs_actions: jax.Array,
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    m_moment: int,
    n_layers: int,
    U_atoms: jax.Array,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
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
    prepare_returns_probability_distribution(
        inputs=inputs,
        weights=weights,
        n_actions=n_actions,
        n_layers=n_layers,
        wires_actions=wiredict['actions'],
        wires_returns=wiredict['returns'],
        pauli_enc=pauli_enc,
        pauli_var=pauli_var,
        imprimitive=imprimitive,
        use_qft=use_qft,
    )
    # Prepare the m-th moment of the returns distribution.
    prepare_m_moment(
        m_moment=m_moment,
        U_atoms=U_atoms,
        wires_returns=wiredict['returns'],
    )
    # Measure the return-atom qubits to collapse their state, then reset to zero.
    circuit_layer_reset(wires=wiredict['returns'])
    
    # Generate action distribuion via measurements across the action wires.
    return qml.probs(wires=wiredict['actions'])

def circuit_return_dist(
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    probs_actions: jax.Array,
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
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
    prepare_returns_probability_distribution(
        inputs=inputs,
        weights=weights,
        n_actions=n_actions,
        n_layers=n_layers,
        wires_actions=wiredict['actions'],
        wires_returns=wiredict['returns'],
        pauli_enc=pauli_enc,
        pauli_var=pauli_var,
        imprimitive=imprimitive,
        use_qft=use_qft,
    )
    
    # Generate weighted return distribuion via measurements across the return wires.
    return qml.probs(wires=wiredict['returns'])

def circuit_return_dist_shifted(
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    probs_actions: jax.Array,
    U_shift: jax.Array, # (2**d_returns, 2**d_returns)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
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
    prepare_returns_probability_distribution(
        inputs=inputs,
        weights=weights,
        n_actions=n_actions,
        n_layers=n_layers,
        wires_actions=wiredict['actions'],
        wires_returns=wiredict['returns'],
        pauli_enc=pauli_enc,
        pauli_var=pauli_var,
        imprimitive=imprimitive,
        use_qft=use_qft,
    )
    
    # Redistribute shifted return-atoms to align with original PMF atoms.
    qml.QubitUnitary(U_shift, wires=wiredict['returns'], unitary_check=False) # We know it's not unitary, so don't check.
    
    # Generate weighted return distribuion via measurements across the return wires.
    return qml.probs(wires=wiredict['returns'])

# TODO Input environment observation must pass through an ML layer to project the observations into a space ingestible for the quantum circuit.
def circuit_hermitian_loss(
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    probs_actions: jax.Array,
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    rho: jax.Array, # Hermitian operator core.
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
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
    prepare_returns_probability_distribution(
        inputs=inputs,
        weights=weights,
        n_actions=n_actions,
        n_layers=n_layers,
        wires_actions=wiredict['actions'],
        wires_returns=wiredict['returns'],
        pauli_enc=pauli_enc,
        pauli_var=pauli_var,
        imprimitive=imprimitive,
        use_qft=use_qft,
    )
    
    # Measure the expectation of the return-atom wires w.r.t. a given Hermitian operator.
    H = qml.Hermitian(rho, wires=wiredict['returns']) # Return-atom wires.
    # H = qml.Hermitian(rho, wires=wirelist) # All wires.
    return qml.expval(H)

def circuit_return_density_matrix(
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    probs_actions: jax.Array,
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
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
    
    qml.Snapshot('after_action_prep', measurement=qml.density_matrix(wires=wirelist))
    
    # Prepare returns probability distribution for every policy.
    prepare_returns_probability_distribution(
        inputs=inputs,
        weights=weights,
        n_actions=n_actions,
        n_layers=n_layers,
        wires_actions=wiredict['actions'],
        wires_returns=wiredict['returns'],
        pauli_enc=pauli_enc,
        pauli_var=pauli_var,
        imprimitive=imprimitive,
        use_qft=use_qft,
    )
    
    qml.Snapshot('after_return_prep', measurement=qml.density_matrix(wires=wirelist))
    
    # Density matrix for return-atom wires.
    return qml.density_matrix(wires=wiredict['returns'])
    # # Density matrix for all wires.
    # return qml.density_matrix(wires=wirelist)

def circuit_return_density_matrix_shifted(
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    probs_actions: jax.Array,
    U_shift: jax.Array, # (2**d_returns, 2**d_returns)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
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
    prepare_returns_probability_distribution(
        inputs=inputs,
        weights=weights,
        n_actions=n_actions,
        n_layers=n_layers,
        wires_actions=wiredict['actions'],
        wires_returns=wiredict['returns'],
        pauli_enc=pauli_enc,
        pauli_var=pauli_var,
        imprimitive=imprimitive,
        use_qft=use_qft,
    )
    
    # Redistribute shifted return-atoms to align with original PMF atoms.
    qml.QubitUnitary(U_shift, wires=wiredict['returns'], unitary_check=False) # We know it's not unitary, so don't check.
    
    # Density matrix for return-atom wires.
    return qml.density_matrix(wires=wiredict['returns'])
    # # Density matrix for all wires.
    # return qml.density_matrix(wires=wirelist)

# @partial(nnx.vmap, in_axes=(0,None,None,None,0,None,0,None,None,None,None,None,None))
# 
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
    'pauli_enc',
    'pauli_var',
    'imprimitive',
    'use_qft',
])
def run_circuit_action_dist(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    m_moment: int,
    n_layers: int,
    U_atoms: jax.Array,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
    ):
    """Run `circuit_action_dist` on a single input instance."""
    device = qml.device(name=device_name, shots=shots, seed=key)
    qnode = qml.QNode(func=circuit_action_dist, device=device, interface="jax", diff_method=diff_method)
    return qnode(
        inputs=inputs,
        probs_actions=probs_actions,
        weights=weights,
        n_actions=n_actions,
        d_actions=d_actions,
        d_returns=d_returns,
        m_moment=m_moment,
        n_layers=n_layers,
        U_atoms=U_atoms,
        pauli_enc=pauli_enc,
        pauli_var=pauli_var,
        imprimitive=imprimitive,
        use_qft=use_qft,
    )

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
    'pauli_enc',
    'pauli_var',
    'imprimitive',
    'use_qft',
])
def batched_run_circuit_action_dist(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    m_moment: int,
    n_layers: int,
    U_atoms: jax.Array,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
    ):
    """Run `circuit_action_dist` on a batch of input instances."""
    # Create vmap of core function.
    mapped_func = nnx.vmap(
        f=run_circuit_action_dist, 
        in_axes=(
            0, # key
            None, # device_name
            None, # shots
            None, # diff_method
            0, # inputs
            None, # weights
            0, # probs_actions
            None, # n_actions
            None, # d_actions
            None, # d_returns
            None, # m_moment
            None, # n_layers
            None, # U_atoms
            None, # pauli_enc
            None, # pauli_var
            None, # imprimitive
            None, # use_qft
            ),
        )
    # Call mapped function with manually assigned positional arguments.
    return mapped_func(
        key,
        device_name,
        shots,
        diff_method,
        inputs,
        weights,
        probs_actions,
        n_actions,
        d_actions,
        d_returns,
        m_moment,
        n_layers,
        U_atoms,
        pauli_enc,
        pauli_var,
        imprimitive,
        use_qft,
    )


@partial(nnx.jit, static_argnames=[
    # Device and QNode parameters.
    'device_name', 'shots', 'diff_method',
    # Circuit parameters.
    'n_actions',
    'd_actions',
    'd_returns',
    'n_layers',
    'pauli_enc',
    'pauli_var',
    'imprimitive',
    'use_qft',
])
def run_circuit_return_dist(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
    ):
    """Run `circuit_return_dist` on a single input instance."""
    device = qml.device(name=device_name, shots=shots, seed=key)
    qnode = qml.QNode(func=circuit_return_dist, device=device, interface="jax", diff_method=diff_method)
    return qnode(
        inputs=inputs,
        probs_actions=probs_actions,
        weights=weights,
        n_actions=n_actions,
        d_actions=d_actions,
        d_returns=d_returns,
        n_layers=n_layers,
        pauli_enc=pauli_enc,
        pauli_var=pauli_var,
        imprimitive=imprimitive,
        use_qft=use_qft,
    )

@partial(nnx.jit, static_argnames=[
    # Device and QNode parameters.
    'device_name', 'shots', 'diff_method',
    # Circuit parameters.
    'n_actions',
    'd_actions',
    'd_returns',
    'n_layers',
    'pauli_enc',
    'pauli_var',
    'imprimitive',
    'use_qft',
])
def batched_run_circuit_return_dist(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
    ):
    """Run `circuit_return_dist` on a batch of input instances."""
    # Create vmap of core function.
    mapped_func = nnx.vmap(
        f=run_circuit_return_dist, 
        in_axes=(
            0, # key
            None, # device_name
            None, # shots
            None, # diff_method
            0, # inputs
            None, # weights
            0, # probs_actions
            None, # n_actions
            None, # d_actions
            None, # d_returns
            None, # n_layers
            None, # pauli_enc
            None, # pauli_var
            None, # imprimitive
            None, # use_qft
            ),
        )
    # Call mapped function with manually assigned positional arguments.
    return mapped_func(
        key,
        device_name,
        shots,
        diff_method,
        inputs,
        weights,
        probs_actions,
        n_actions,
        d_actions,
        d_returns,
        n_layers,
        pauli_enc,
        pauli_var,
        imprimitive,
        use_qft,
    )

@partial(nnx.jit, static_argnames=[
    # Device and QNode parameters.
    'device_name', 'shots', 'diff_method',
    # Circuit parameters.
    'n_actions',
    'd_actions',
    'd_returns',
    'n_layers',
    'pauli_enc',
    'pauli_var',
    'imprimitive',
    'use_qft',
])
def run_circuit_return_dist_shifted(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    probs_actions: jax.Array,
    U_shift: jax.Array, # (2**d_returns,2**d_returns,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
    ):
    """Run `circuit_return_dist` on a single input instance."""
    device = qml.device(name=device_name, shots=shots, seed=key)
    qnode = qml.QNode(func=circuit_return_dist_shifted, device=device, interface="jax", diff_method=diff_method)
    return qnode(
        inputs=inputs,
        probs_actions=probs_actions,
        U_shift=U_shift,
        weights=weights,
        n_actions=n_actions,
        d_actions=d_actions,
        d_returns=d_returns,
        n_layers=n_layers,
        pauli_enc=pauli_enc,
        pauli_var=pauli_var,
        imprimitive=imprimitive,
        use_qft=use_qft,
    )

@partial(nnx.jit, static_argnames=[
    # Device and QNode parameters.
    'device_name', 'shots', 'diff_method',
    # Circuit parameters.
    'n_actions',
    'd_actions',
    'd_returns',
    'n_layers',
    'pauli_enc',
    'pauli_var',
    'imprimitive',
    'use_qft',
])
def batched_run_circuit_return_dist_shifted(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    probs_actions: jax.Array,
    U_shift: jax.Array, # (2**d_returns,2**d_returns,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
    ):
    """Run `circuit_return_dist` on a batch of input instances."""
    # Create vmap of core function.
    mapped_func = nnx.vmap(
        f=run_circuit_return_dist_shifted, 
        in_axes=(
            0, # key
            None, # device_name
            None, # shots
            None, # diff_method
            0, # inputs
            None, # weights
            0, # probs_actions
            0, # U_shift
            None, # n_actions
            None, # d_actions
            None, # d_returns
            None, # n_layers
            None, # pauli_enc
            None, # pauli_var
            None, # imprimitive
            None, # use_qft
            ),
        )
    # Call mapped function with manually assigned positional arguments.
    return mapped_func(
        key,
        device_name,
        shots,
        diff_method,
        inputs,
        weights,
        probs_actions,
        U_shift,
        n_actions,
        d_actions,
        d_returns,
        n_layers,
        pauli_enc,
        pauli_var,
        imprimitive,
        use_qft,
    )


@partial(nnx.jit, static_argnames=[
    # Device and QNode parameters.
    'device_name', 'shots', 'diff_method',
    # Circuit parameters.
    'n_actions',
    'd_actions',
    'd_returns',
    'n_layers',
    'pauli_enc',
    'pauli_var',
    'imprimitive',
    'use_qft',
])
def run_circuit_hermitian_loss(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    rho: jax.Array, # Hermitian operator core.
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
    ):
    """Run `circuit_hermitian_loss` on a single input instance."""
    device = qml.device(name=device_name, shots=shots, seed=key)
    qnode = qml.QNode(func=circuit_hermitian_loss, device=device, interface="jax", diff_method=diff_method)
    return qnode(
        inputs=inputs,
        probs_actions=probs_actions,
        weights=weights,
        n_actions=n_actions,
        d_actions=d_actions,
        d_returns=d_returns,
        n_layers=n_layers,
        rho=rho,
        pauli_enc=pauli_enc,
        pauli_var=pauli_var,
        imprimitive=imprimitive,
        use_qft=use_qft,
    )

@partial(nnx.jit, static_argnames=[
    # Device and QNode parameters.
    'device_name', 'shots', 'diff_method',
    # Circuit parameters.
    'n_actions',
    'd_actions',
    'd_returns',
    'n_layers',
    'pauli_enc',
    'pauli_var',
    'imprimitive',
    'use_qft',
])
def batched_run_circuit_hermitian_loss(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    rho: jax.Array, # Hermitian operator core.
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
    ):
    """Run `circuit_hermitian_loss` on a batch of input instances."""
    # Create vmap of core function.
    mapped_func = nnx.vmap(
        f=run_circuit_hermitian_loss, 
        in_axes=(
            0, # key
            None, # device_name
            None, # shots
            None, # diff_method
            0, # inputs
            None, # weights
            0, # probs_actions
            None, # n_actions
            None, # d_actions
            None, # d_returns
            None, # n_layers
            0, # rho
            None, # pauli_enc
            None, # pauli_var
            None, # imprimitive
            None, # use_qft
            ),
        )
    # Call mapped function with manually assigned positional arguments.
    return mapped_func(
        key,
        device_name,
        shots,
        diff_method,
        inputs,
        weights,
        probs_actions,
        n_actions,
        d_actions,
        d_returns,
        n_layers,
        rho,
        pauli_enc,
        pauli_var,
        imprimitive,
        use_qft,
    )


@partial(nnx.jit, static_argnames=[
    # Device and QNode parameters.
    'device_name', 'shots', 'diff_method',
    # Circuit parameters.
    'n_actions',
    'd_actions',
    'd_returns',
    'n_layers',
    'pauli_enc',
    'pauli_var',
    'imprimitive',
    'use_qft',
])
def run_circuit_return_density_matrix(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
    ):
    """Run `circuit_return_density_matrix` on a single of input instance."""
    device = qml.device(name=device_name, shots=shots, seed=key)
    qnode = qml.QNode(func=circuit_return_density_matrix, device=device, interface="jax", diff_method=diff_method)
    return qnode(
        inputs=inputs,
        probs_actions=probs_actions,
        weights=weights,
        n_actions=n_actions,
        d_actions=d_actions,
        d_returns=d_returns,
        n_layers=n_layers,
        pauli_enc=pauli_enc,
        pauli_var=pauli_var,
        imprimitive=imprimitive,
        use_qft=use_qft,
    )

@partial(nnx.jit, static_argnames=[
    # Device and QNode parameters.
    'device_name', 'shots', 'diff_method',
    # Circuit parameters.
    'n_actions',
    'd_actions',
    'd_returns',
    'n_layers',
    'pauli_enc',
    'pauli_var',
    'imprimitive',
    'use_qft',
])
def batched_run_circuit_return_density_matrix(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
    ):
    """Run `circuit_return_density_matrix` on a batch of input instances."""
    # Create vmap of core function.
    mapped_func = nnx.vmap(
        f=run_circuit_return_density_matrix, 
        in_axes=(
            0, # key
            None, # device_name
            None, # shots
            None, # diff_method
            0, # inputs
            None, # weights
            0, # probs_actions
            None, # n_actions
            None, # d_actions
            None, # d_returns
            None, # n_layers
            None, # pauli_enc
            None, # pauli_var
            None, # imprimitive
            None, # use_qft
            ),
        )
    # Call mapped function with manually assigned positional arguments.
    return mapped_func(
        key,
        device_name,
        shots,
        diff_method,
        inputs,
        weights,
        probs_actions,
        n_actions,
        d_actions,
        d_returns,
        n_layers,
        pauli_enc,
        pauli_var,
        imprimitive,
        use_qft,
    )


@partial(nnx.jit, static_argnames=[
    # Device and QNode parameters.
    'device_name', 'shots', 'diff_method',
    # Circuit parameters.
    'n_actions',
    'd_actions',
    'd_returns',
    'n_layers',
    'pauli_enc',
    'pauli_var',
    'imprimitive',
    'use_qft',
])
def run_circuit_return_density_matrix_shifted(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    probs_actions: jax.Array,
    U_shift: jax.Array, # (2**d_returns,2**d_returns,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
    ):
    """Run `circuit_return_density_matrix_shifted` on a single of input instance."""
    device = qml.device(name=device_name, shots=shots, seed=key)
    qnode = qml.QNode(func=circuit_return_density_matrix_shifted, device=device, interface="jax", diff_method=diff_method)
    return qnode(
        inputs=inputs,
        probs_actions=probs_actions,
        U_shift=U_shift,
        weights=weights,
        n_actions=n_actions,
        d_actions=d_actions,
        d_returns=d_returns,
        n_layers=n_layers,
        pauli_enc=pauli_enc,
        pauli_var=pauli_var,
        imprimitive=imprimitive,
        use_qft=use_qft,
    )

@partial(nnx.jit, static_argnames=[
    # Device and QNode parameters.
    'device_name', 'shots', 'diff_method',
    # Circuit parameters.
    'n_actions',
    'd_actions',
    'd_returns',
    'n_layers',
    'pauli_enc',
    'pauli_var',
    'imprimitive',
    'use_qft',
])
def batched_run_circuit_return_density_matrix_shifted(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, num_enc_features,)
    weights: jax.Array, # (n_actions, n_layers, d_returns, num_var_features,)
    probs_actions: jax.Array,
    U_shift: jax.Array, # (2**d_returns,2**d_returns,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    imprimitive: str | None, # Entanglement gate string.
    use_qft: bool,
    ):
    """Run `circuit_return_density_matrix_shifted` on a batch of input instances."""
    # Create vmap of core function.
    mapped_func = nnx.vmap(
        f=run_circuit_return_density_matrix_shifted, 
        in_axes=(
            0, # key
            None, # device_name
            None, # shots
            None, # diff_method
            0, # inputs
            None, # weights
            0, # probs_actions
            0, # U_shift
            None, # n_actions
            None, # d_actions
            None, # d_returns
            None, # n_layers
            None, # pauli_enc
            None, # pauli_var
            None, # imprimitive
            None, # use_qft
            ),
        )
    # Call mapped function with manually assigned positional arguments.
    return mapped_func(
        key,
        device_name,
        shots,
        diff_method,
        inputs,
        weights,
        probs_actions,
        U_shift,
        n_actions,
        d_actions,
        d_returns,
        n_layers,
        pauli_enc,
        pauli_var,
        imprimitive,
        use_qft,
    )



@dataclass(kw_only=True)
class QuantumNativeQNetwork(SerializableModule):
    """Quantum-native (distributional) Q-value network."""
    din: int
    """Dimension of input data. The model will map this dimension into one that is supported by the number of return-atom qubits."""
    n_actions: int
    """Number of actions (will require `ceil(log2(n))` qubits)."""
    n_returns: int
    """Number of return atoms (will require `ceil(log2(n))` qubits)."""
    z_min: float
    """Lower bound for return value."""
    z_max: float
    """Upper bound for return value."""
    m_moment: int
    """m-th moment of the return atoms distribution (`E[Z^m]`)."""
    n_reuploads: int
    """n-th power of the expectation over the return atoms distribution (`E[Z]^n`)."""
    n_layers: int
    """Number of re-upload layers to use for the return-atom distribution unitary."""
    rngs_kwargs: dict[str, int]
    pauli_enc: tuple[str, ...]
    """List of Pauli operator strings for encoding layer (e.g., 'X', 'Y', 'Z', 'RX', 'RY', 'RZ')."""
    pauli_var: tuple[str, ...]
    """List of Pauli operator strings for variational layer (e.g., 'X', 'Y', 'Z', 'RX', 'RY', 'RZ')."""
    imprimitive: str | None
    """Entanglement gate string (e.g., 'CNOT', 'CX', 'CZ')."""
    use_qft: bool = True # Default to True to uphold previous behavior.
    """Use a quantum Fourier transform (QFT) at the end of the return distribution gates."""
    input_preprocess: SerializableModule | None = None
    """Nested module for input preprocessing (can be trainable). We need this because the `QuantumNativeQNetworkSplit` class has methods required by the QnRL training algorithm, hence it cannot be wrapped inside another module. To support customizing input preprocessing on a per-environment basis we allow a nested input preprocessing layer to be a member of the `QuantumNativeQNetworkSplit` module."""
    input_multiplex_activation: str = 'linear'
    """Activation function to use after multiplexing input for quantum circuit. Must be a string key contained within the `qnrl.modules.serializable.activation_function_dispatcher` dictionary."""
    
    def __post_init__(self):
        
        # Compute qubit count based on number of policies, actions, and return atoms.
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
        self.atoms = jnp.array(np.linspace(start=self.z_min, stop=self.z_max, num=self.n_returns))
        # Create non-linear trace non-increasing Kraus matrix of returns atoms.
        atoms_values = self.atoms
        atoms_values = atoms_values - atoms_values.min() # Start from zero.
        atoms_values = atoms_values / atoms_values.sum() # Normalize to probability distribution.
        atoms_values = jnp.sqrt(atoms_values) # Ensure that L2-norm of quantum state amplitude will produce the atom probability (i.e., $|amp|^2 = prob$).
        # Pad with zeros if return atoms shape is not a power of 2.
        if self.n_returns != 2**self.d_returns:
            atoms_values = jnp.pad(atoms_values, (0,2**self.d_returns - self.n_returns), mode='constant')
        # Create quantum operator from the atom values.
        self.U_atoms = jnp.diag(atoms_values)
        
        # Ensure input multiplex activation string key is valid.
        assert self.input_multiplex_activation in activation_function_dispatcher, f"Invalid activation function key {self.input_multiplex_activation}"
        
        # Create parameters that depend on random seed.
        self.reseed(self.rngs_kwargs)
    
    def pre_reseed(self, rngs_kwargs: dict[str, int]):
        # Weights for return-atom distribution generation.
        # Every action gets their own return-atom distribution unitary.
        num_var_params = get_ops_num_params(ops=self.pauli_var)
        self.circ_weights_var_shape = (self.n_actions,self.n_layers,self.d_returns,num_var_params,) # Each qubit has (Rz, Ry, Rz).
        
        # Inputs to circuit will be encoded onto the return-atom wires.
        num_enc_params = get_ops_num_params(ops=self.pauli_enc)
        self.circ_weights_enc_obs_shape = (self.n_actions,self.n_layers,self.d_returns,num_enc_params,)
        self.circ_observation_shape_in = self.din # Dimension of the input data.
        self.circ_observation_shape_out = (self.n_actions,self.d_returns,num_enc_params,)
    
    def post_reseed(self, rngs_kwargs: dict[str, int]):
        self.circ_weights_var = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_var_shape, minval=0., maxval=jnp.pi))
        
        self.circ_weights_enc_obs_kernel = nnx.Param(nnx.initializers.ones_init()(key=self.rngs.params(), shape=self.circ_weights_enc_obs_shape)) # w
        # self.circ_weights_enc_obs_bias = nnx.Param(nnx.initializers.zeros_init()(key=self.rngs.params(), shape=self.circ_weights_enc_obs_shape)) # b
        
        # Projects input dimension to dimension supported by quantum circuit.
        self.linear_input_projection = nnx.Linear(in_features=self.circ_observation_shape_in, out_features=reduce(lambda x,y: x*y, self.circ_observation_shape_out), rngs=self.rngs, param_dtype='float64')
    
    def reseed(self, rngs_kwargs: dict[str, int]):
        """Re-create model parameters using a dictionary of `nnx.Rngs` keyword arguments."""
        self.pre_reseed(rngs_kwargs=rngs_kwargs)
        self.rngs_kwargs = rngs_kwargs
        self.rngs: nnx.Rngs = nnx.Rngs(**self.rngs_kwargs)
        self.post_reseed(rngs_kwargs=rngs_kwargs)
    
    @nnx.jit
    def __call__(self, x): # (batch,din,)
        return self.call_gen_action_dist(x) # Generation action distribution.
    
    @nnx.jit
    def pre_project_observations(self, x: jax.Array): # (batch,din,)
        """Handle all preprocessing of input observations prior to projection into quantum circuit."""
        # Preprocess input data.
        if self.input_preprocess is not None:
            x = self.input_preprocess(x)
        return x

    @nnx.jit
    def post_project_observations(self, x: jax.Array): # (batch,action,qubit,feat,)
        """Handle multiplexing the projected input observations into the quantum circuit."""
        x = jnp.reshape(x, (-1, *self.circ_observation_shape_out,))
        # Multiplex the encoded observations across the layers of each action operator.
        # Einsum dimension labels:
        #   b = batch
        #   l = layer
        #   q = qubit
        #   f = feature
        x = jnp.einsum("alqf,baqf->balqf", self.circ_weights_enc_obs_kernel, x)
        # Pass multiplex result through activation function.
        x = activation_function_dispatcher[self.input_multiplex_activation](x)
        return x
    
    @nnx.jit
    def project_observations(self, x: jax.Array): # (batch,din,)
        x = self.pre_project_observations(x) # Preprocess input observations.
        ###
        x = self.linear_input_projection(x) # Project input data into dimension required for quantum circuit.
        ###
        x = self.post_project_observations(x) # Multiplex projected observations into quantum circuit.
        return x
    
    @nnx.jit(static_argnames=['n_reuploads'])
    def call_gen_action_dist(self, x, n_reuploads: int | None = None): # (batch,din,)
        """Generate action distribuion from input observation."""
        
        # Project input observation into shape that is compatible with the quantum circuit.
        x = self.project_observations(x) # (batch, n_layers, d_returns, 1,)
        
        # Track the batch size.
        batch_size = x.shape[0]
        
        # Prepare the n-th power of the m-th moment of the returns distribution.
        key = self.rngs.default() # Generate new random key for circuit measurement.
        # Initially set all actions to equal probability (here all have magnitude of 1, which will be normalized within the quantum circuit code later).
        probs_actions = jnp.ones(shape=(batch_size, self.n_actions,), dtype='float64')
        if n_reuploads is None:
            n_reuploads = self.n_reuploads
        # Create a batch of keys.
        subkeys = [key]
        for _ in range(n_reuploads):
            # Create a batch of keys for the current reupload.
            subkeys = jax.random.split(key=subkeys[0], num=batch_size)
            # Run the circuit on batched input.
            probs_actions = batched_run_circuit_action_dist( # (batch,2**d_actions,)
                # All device and QNode parameters.
                key=subkeys,
                device_name="default.qubit",
                shots=None,
                diff_method=None,
                # Circuit arguments.
                inputs=x,
                probs_actions=probs_actions,
                weights=self.circ_weights_var,
                n_actions=self.n_actions,
                d_actions=self.d_actions,
                d_returns=self.d_returns,
                m_moment=self.m_moment,
                n_layers=self.n_layers,
                U_atoms=self.U_atoms,
                pauli_enc=self.pauli_enc,
                pauli_var=self.pauli_var,
                imprimitive=self.imprimitive,
                use_qft=self.use_qft,
            )
            # Filter circuit output based on number of "active" actions.
            # If desired number of actions is not a multiple of 2, then filter out the excess actions on the qubit register (i.e., more actions can be represented on the register and we don't care about them).
            probs_actions = probs_actions[:,jnp.array(self.active_action_idx, dtype=int)] # (batch,n_actions,)
            # Ensure probabilities are normalized after the filter.
            probs_actions = probs_actions / probs_actions.sum(axis=-1, keepdims=True)
        return probs_actions
    
    @nnx.jit
    def call_gen_return_dist(self, x, probs_actions): # (batch,din,)
        """Generate return distribuion from input observation."""
        
        # Project input observation into shape that is compatible with the quantum circuit.
        x = self.project_observations(x) # (batch, n_layers, d_returns, 1,)
        
        # Track the batch size.
        batch_size = x.shape[0]
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()

        # Create a batch of keys.
        subkeys = jax.random.split(key=key, num=batch_size)
        
        # Run the circuit.
        probs_returns = batched_run_circuit_return_dist( # (batch,2**d_returns,)
            # All device and QNode parameters.
            key=subkeys,
            device_name='default.qubit',
            shots=None,
            diff_method='backprop',
            #
            # Circuit arguments.
            inputs=x,
            probs_actions=probs_actions,
            weights=self.circ_weights_var,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            n_layers=self.n_layers,
            pauli_enc=self.pauli_enc,
            pauli_var=self.pauli_var,
            imprimitive=self.imprimitive,
            use_qft=self.use_qft,
        )
        # Filter circuit output based on number of "active" returns.
        # If desired number of returns is not a multiple of 2, then filter out the excess returns on the qubit register (i.e., more returns can be represented on the register and we don't care about them).
        probs_returns = probs_returns[:,jnp.array(self.active_return_idx, dtype=int)] # (batch,n_returns,)
        # Ensure probabilities are normalized after the filter.
        probs_returns = probs_returns / probs_returns.sum(axis=-1, keepdims=True)
        return probs_returns
    
    @nnx.jit
    def call_gen_return_dist_shifted(self, x, probs_actions, U_shift): # (batch,din,)
        """Generate shifted return distribuion from input observation and action probability distribution."""
        
        # Project input observation into shape that is compatible with the quantum circuit.
        x = self.project_observations(x) # (batch, n_layers, d_returns, 1,)
        
        # Track the batch size.
        batch_size = x.shape[0]
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()

        # Create a batch of keys.
        subkeys = jax.random.split(key=key, num=batch_size)
        
        # Run the circuit.
        probs_returns = batched_run_circuit_return_dist_shifted( # (batch,2**d_returns,)
            # All device and QNode parameters.
            key=subkeys,
            device_name='default.mixed',
            shots=None,
            diff_method='backprop',
            #
            # Circuit arguments.
            inputs=x,
            probs_actions=probs_actions,
            U_shift=U_shift,
            weights=self.circ_weights_var,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            n_layers=self.n_layers,
            pauli_enc=self.pauli_enc,
            pauli_var=self.pauli_var,
            imprimitive=self.imprimitive,
            use_qft=self.use_qft,
        )
        # Filter circuit output based on number of "active" returns.
        # If desired number of returns is not a multiple of 2, then filter out the excess returns on the qubit register (i.e., more returns can be represented on the register and we don't care about them).
        probs_returns = probs_returns[:,jnp.array(self.active_return_idx, dtype=int)] # (batch,n_returns,)
        # Ensure probabilities are normalized after the filter.
        probs_returns = probs_returns / probs_returns.sum(axis=-1, keepdims=True)
        return probs_returns
    
    
    @nnx.jit
    def call_hermitian_loss(self, x, probs_actions, rho): # (batch,din,)
        """Compute Hermitian loss across the return-atom qubits with respect to an input density matrix rho."""
        
        # Project input observation into shape that is compatible with the quantum circuit.
        x = self.project_observations(x) # (batch, n_layers, d_returns, 1,)
        
        # Track the batch size.
        batch_size = x.shape[0]
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()

        # Create a batch of keys.
        subkeys = jax.random.split(key=key, num=batch_size)
        
        # Run the circuit.
        loss = batched_run_circuit_hermitian_loss( # (batch,2**d_returns,)
            # All device and QNode parameters.
            key=subkeys,
            device_name="default.qubit",
            shots=None,
            diff_method='backprop',
            # Circuit arguments.
            inputs=x,
            probs_actions=probs_actions,
            weights=self.circ_weights_var,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            n_layers=self.n_layers,
            rho=rho,
            pauli_enc=self.pauli_enc,
            pauli_var=self.pauli_var,
            imprimitive=self.imprimitive,
            use_qft=self.use_qft,
        )
        return loss
    
    @nnx.jit
    def call_return_density_matrix(self, x, probs_actions): # (batch,din,)
        """Generate the density matrix across the return-atom qubits according to an input observation and action probability distribution."""
        
        # Project input observation into shape that is compatible with the quantum circuit.
        x = self.project_observations(x) # (batch, n_layers, d_returns, 1,)
        
        # Track the batch size.
        batch_size = x.shape[0]
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()

        # Create a batch of keys.
        subkeys = jax.random.split(key=key, num=batch_size)
        
        # Run the circuit.
        rho = batched_run_circuit_return_density_matrix( # (batch,2**d_returns,2**d_returns)
            # All device and QNode parameters.
            key=subkeys,
            device_name="default.qubit", # Use mixed state simulation for density matrix.
            shots=None,
            diff_method='backprop',
            # Circuit arguments.
            inputs=x,
            probs_actions=probs_actions,
            weights=self.circ_weights_var,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            n_layers=self.n_layers,
            pauli_enc=self.pauli_enc,
            pauli_var=self.pauli_var,
            imprimitive=self.imprimitive,
            use_qft=self.use_qft,
        )
        return rho
    
    @nnx.jit
    def call_return_density_matrix_shifted(self, x, probs_actions, U_shift): # (batch,din,)
        """Generate the shifted density matrix across the return-atom qubits according to an input observation and action probability distribution."""
        
        # Project input observation into shape that is compatible with the quantum circuit.
        x = self.project_observations(x) # (batch, n_layers, d_returns, 1,)
        
        # Track the batch size.
        batch_size = x.shape[0]
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()

        # Create a batch of keys.
        subkeys = jax.random.split(key=key, num=batch_size)
        
        # Run the circuit.
        rho = batched_run_circuit_return_density_matrix_shifted( # (batch,2**d_returns,2**d_returns)
            # All device and QNode parameters.
            key=subkeys,
            device_name="default.qubit", # Use mixed state simulation for density matrix.
            shots=None,
            diff_method='backprop',
            # Circuit arguments.
            inputs=x,
            probs_actions=probs_actions,
            U_shift=U_shift,
            weights=self.circ_weights_var,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            n_layers=self.n_layers,
            pauli_enc=self.pauli_enc,
            pauli_var=self.pauli_var,
            imprimitive=self.imprimitive,
            use_qft=self.use_qft,
        )
        return rho