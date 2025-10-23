from dataclasses import dataclass
from functools import partial, reduce
from itertools import combinations
from typing import Literal

import pennylane as qml
import numpy as np
import jax
from jax import numpy as jnp # JAX version of NumPy.
from flax import nnx # Neural networks in JAX.

from .serializable import SerializableModule


def circuit_layer_circ(wires: list[int], op: Literal['cz', 'cnot', 'cx']):
    """Circular entanglement layer using `CZ`.

    Args:
        wires (list[int]): List of wires.
    """
    match op.lower():
        case 'cz':
            op_func = qml.CZ
        case 'cnot' | 'cx':
            op_func = qml.CNOT
        case _:
            raise ValueError(f"Unsupported entanglement operation: {op}")
    
    for w0, w1 in zip(wires, wires[1:]):
        op_func(wires=[w0, w1]) # Control w0, target w1.
    if len(wires) != 2:
        # op_func(wires=[wires[0], wires[-1]]) # Control w[0], target w[-1].
        op_func(wires=[wires[-1], wires[0]]) # Control w[-1], target w[0].

# def circuit_layer_enc(phi: jax.Array, wires: list[int]):
#     """Encoding layer using `RX`.

#     Args:
#         phi (jax.Array): Encoding parameters for the layer of shape `(num_wires,1)`.
#         wires (list[int]): _description_
#     """
#     for w in wires:
#         # ## Original
#         # qml.RX(phi=phi[w,0], wires=w)
#         ####
#         qml.Hadamard(wires=w)
#         qml.PhaseShift(phi=2*phi[w,0], wires=w)

def circuit_layer_enc(phi: jax.Array, wires: list[int], pauli: tuple[str, ...] = ('Z',)):
    """Encoding layer using Pauli feature map.

    Args:
        phi (jax.Array): Encoding parameters for the layer of shape `(num_wires,1)`.
        wires (list[int]): _description_
    """
    
    def basis_change(wire_tup: tuple[int, ...], pauli_tup: tuple[str, ...], inverse: bool = False):
        assert len(wire_tup) == len(pauli_tup), "number of wires must match length of Pauli string"
        ## Basis change.
        # Iterate over every wire in the n-pair, and match it with associate Pauli.
        for w, p in zip(wire_tup, pauli_tup):
            match (p, inverse):
                case ('x', _):
                    qml.Hadamard(wires=w)
                case ('y', False):
                    qml.SX(wires=w)
                case ('y', True):
                    qml.adjoint(qml.SX(wires=w))
    
    def cx_chain(wire_tup: tuple[int, ...], inverse: bool = False):
        ## CNOT chain.
        # Pair items in the tuple in groups of 2.
        gen = range(len(wire_tup) - 1)
        if inverse:
            gen = reversed(gen)
        for i in gen:
            qml.CNOT(wires=[wire_tup[i], wire_tup[i+1]])
    
    # print(f"circuit_layer_enc: {phi.shape = }")
    # jax.debug.print("circuit_layer_enc: {}", phi)
    for f, pauli_string in enumerate(pauli): # (feature_index, pauli_string)
        match pauli_string.lower():
            case 'rx':
                for w in wires:
                    qml.RX(phi=phi[w,f], wires=w)
            case 'ry':
                for w in wires:
                    qml.RY(phi=phi[w,f], wires=w)
            case 'rz':
                for w in wires:
                    qml.RZ(phi=phi[w,f], wires=w)
            case _:
                # Apply Hadamard on every qubit.
                for w in wires:
                    qml.Hadamard(wires=w)
                
                # Collect the individual Pauli operators of the current Pauli string chain.
                # Reverse the order for correct qubit assignment (i.e., [p2,p1,p0][::-1] -> [q0,q1,q2]).
                pauli_tup = tuple(pauli_string.lower()[::-1])
                # Iterate over all n-pairs of wires.
                wire_pairs = list(combinations(wires, len(pauli_tup)))
                for wire_tup in wire_pairs:
                    ## Basis change.
                    basis_change(wire_tup=wire_tup, pauli_tup=pauli_tup, inverse=False)
                    ## CNOT chain.
                    cx_chain(wire_tup=wire_tup, inverse=False)
                    ## Phase gate at last qubit in the tuple chain.
                    # Rotation angle is only a single parameter.
                    if len(wire_tup) == 1:
                        angle = phi[wire_tup[-1],f]
                    # Rotation angle is a difference of multiple parameters in the Pauli chain.
                    else:
                        angle = reduce(lambda a, b: a * b, jnp.pi - phi[wire_tup,f])
                    # # Multiply angle by 2.
                    # angle = angle * 2.
                    # Apply phase shift using custom rotation angle.
                    qml.PhaseShift(phi=angle, wires=wire_tup[-1])
                    ## CNOT chain.
                    cx_chain(wire_tup=wire_tup, inverse=True)
                    ## Basis change.
                    basis_change(wire_tup=wire_tup, pauli_tup=pauli_tup, inverse=True)


def circuit_layer_var(phi: jax.Array, wires: list[int], pauli: tuple[str,...]):
    """Variational layer `RZ * RY * RZ |psi>`.

    Args:
        phi (jax.Array): Variational parameters for the layer of shape `(num_wires,3)`.
        wires (list[int]): List of wires.
    """

    # print(f"circuit_layer_var: {phi.shape = }")
    for f, pauli_string in enumerate(pauli): # (feature_index, pauli_string)
        match pauli_string.lower():
            case 'rx':
                for w in wires:
                    qml.RX(phi=phi[w,f], wires=w)
            case 'ry':
                for w in wires:
                    qml.RY(phi=phi[w,f], wires=w)
            case 'rz':
                for w in wires:
                    qml.RZ(phi=phi[w,f], wires=w)
            case 'globalphase':
                for w in wires:
                    qml.GlobalPhase(phi=phi[w,f], wires=w)
            case _:
                raise ValueError(f"Unsupported Pauli string: {pauli_string}")

def circuit_layer_reset(wires: list[int]):
    """Measure the given qubits to collapse their state, then reset to zero."""
    for w in wires:
        qml.measure(wires=w, reset=True)

def get_ops_for_returns_distribution(
    inputs: jax.Array, # (n_layers, d_returns, 1,)
    weights: jax.Array, # (n_layers+1, d_returns, 3,)
    n_layers: int,
    wires: list[int],
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    ):
    """Return-atom distribution unitary.
    
    Generates a return distribuion encoded on the given wires.
    
    The schema of this operation is:
        - `n_layers+1` Variational layers
        - `n_layers` Circular-entanglement layers
        - `n_layers` Input encoding layers
    """
    
    # # Apply Hadamard on every qubit.
    # for w in wires:
    #     qml.Hadamard(wires=w)
    
    for l in range(n_layers):
        circuit_layer_enc(phi=inputs[l], wires=wires, pauli=pauli_enc)
        # circuit_layer_enc(phi=inputs, wires=wires, pauli=pauli_enc)
        circuit_layer_var(phi=weights[l], wires=wires, pauli=pauli_var)
        # qml.StronglyEntanglingLayers(weights=weights[l], wires=wires) # CNOT entanglement by default.
        # qml.StronglyEntanglingLayers(weights=weights[l], wires=wires, imprimitive=qml.CZ) # Turn off entanglement.
        # qml.StronglyEntanglingLayers(weights=weights[l], wires=wires, imprimitive=qml.Identity) # Turn off entanglement.
        # circuit_layer_circ(wires=wires, op='cnot')
        # circuit_layer_circ(wires=wires, op='cz')
        # circuit_layer_enc(phi=inputs, wires=wires, pauli=pauli_enc)
        ########
        # for w in wires:
        #     qml.Hadamard(wires=w)
        # circuit_layer_enc(phi=inputs[l], wires=wires, pauli=pauli_enc)
        # circuit_layer_circ(wires=wires, op='cnot')
        # circuit_layer_var(phi=weights[l], wires=wires, pauli=pauli_var)
        #######
        # circuit_layer_enc(phi=inputs[l], wires=wires, pauli=pauli)
        # circuit_layer_var(phi=weights[l], wires=wires)
        # circuit_layer_circ(wires=wires)
        ########
        # circuit_layer_var(phi=weights[l], wires=wires)
        # circuit_layer_circ(wires=wires)
        # circuit_layer_enc(phi=inputs[l], wires=wires, pauli=pauli)
    #
    #######
    # circuit_layer_var(phi=weights[-1], wires=wires, pauli=pauli_var) # Final variational layer at the end.
    # circuit_layer_circ(wires=wires, op='cnot')
    # circuit_layer_circ(wires=wires, op='cz')
    #######
    # circuit_layer_circ(wires=wires, op='cnot')
    # circuit_layer_var(phi=weights[-1], wires=wires, pauli=pauli_var) # Final variational layer at the end.
    
    # 
    # circuit_layer_enc(phi=inputs[-1], wires=wires, pauli=pauli)
    # ########
    # for l in range(n_layers):
    #     circuit_layer_enc(phi=inputs[l], wires=wires, pauli=pauli)
    #     circuit_layer_circ(wires=wires)

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

def prepare_returns_probability_distribution(
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    n_actions: int,
    n_layers: int,
    wires_actions: list[int],
    wires_returns: list[int],
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
    ):
    """Prepare return probability distribuion for every action."""
    ops = [
        qml.prod(get_ops_for_returns_distribution)(
            # inputs=inputs,
            inputs=inputs[a],
            weights=weights[a],
            n_layers=n_layers,
            wires=wires_returns,
            pauli_enc=pauli_enc,
            pauli_var=pauli_var,
        )
        for a in range(n_actions)
    ]
    qml.Select(ops, control=wires_actions)

def circuit_action_dist(
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    probs_actions: jax.Array,
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    m_moment: int,
    n_layers: int,
    U_atoms: jax.Array,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    probs_actions: jax.Array,
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
    )
    
    # Generate weighted return distribuion via measurements across the return wires.
    return qml.probs(wires=wiredict['returns'])

def circuit_return_dist_shifted(
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    probs_actions: jax.Array,
    U_shift: jax.Array, # (2**d_returns, 2**d_returns)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
    )
    
    # Redistribute shifted return-atoms to align with original PMF atoms.
    qml.QubitUnitary(U_shift, wires=wiredict['returns'], unitary_check=False) # We know it's not unitary, so don't check.
    
    # Generate weighted return distribuion via measurements across the return wires.
    return qml.probs(wires=wiredict['returns'])

# TODO Input environment observation must pass through an ML layer to project the observations into a space ingestible for the quantum circuit.
def circuit_hermitian_loss(
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    probs_actions: jax.Array,
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    rho: jax.Array, # Hermitian operator core.
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
    )
    
    # Measure the expectation of the return-atom wires w.r.t. a given Hermitian operator.
    H = qml.Hermitian(rho, wires=wiredict['returns']) # Return-atom wires.
    # H = qml.Hermitian(rho, wires=wirelist) # All wires.
    return qml.expval(H)

def circuit_return_density_matrix(
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    probs_actions: jax.Array,
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
    )
    
    qml.Snapshot('after_return_prep', measurement=qml.density_matrix(wires=wirelist))
    
    # Density matrix for return-atom wires.
    return qml.density_matrix(wires=wiredict['returns'])
    # # Density matrix for all wires.
    # return qml.density_matrix(wires=wirelist)

def circuit_return_density_matrix_shifted(
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    probs_actions: jax.Array,
    U_shift: jax.Array, # (2**d_returns, 2**d_returns)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
])
def run_circuit_action_dist(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    m_moment: int,
    n_layers: int,
    U_atoms: jax.Array,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
])
def batched_run_circuit_action_dist(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    m_moment: int,
    n_layers: int,
    U_atoms: jax.Array,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
])
def run_circuit_return_dist(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
])
def batched_run_circuit_return_dist(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
])
def run_circuit_return_dist_shifted(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    probs_actions: jax.Array,
    U_shift: jax.Array, # (2**d_returns,2**d_returns,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
])
def batched_run_circuit_return_dist_shifted(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    probs_actions: jax.Array,
    U_shift: jax.Array, # (2**d_returns,2**d_returns,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
])
def run_circuit_hermitian_loss(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    rho: jax.Array, # Hermitian operator core.
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
])
def batched_run_circuit_hermitian_loss(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    rho: jax.Array, # Hermitian operator core.
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
])
def run_circuit_return_density_matrix(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
])
def batched_run_circuit_return_density_matrix(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    probs_actions: jax.Array,
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
])
def run_circuit_return_density_matrix_shifted(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    probs_actions: jax.Array,
    U_shift: jax.Array, # (2**d_returns,2**d_returns,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
])
def batched_run_circuit_return_density_matrix_shifted(
    # All device and QNode parameters.
    key: jax.Array,
    device_name: str,
    shots: int | None,
    diff_method: str | None,
    # Circuit arguments.
    inputs: jax.Array, # (n_actions, n_layers, d_returns, 1,)
    weights: jax.Array, # (n_actions, n_layers+1, d_returns, 3,)
    probs_actions: jax.Array,
    U_shift: jax.Array, # (2**d_returns,2**d_returns,)
    n_actions: int,
    d_actions: int,
    d_returns: int,
    n_layers: int,
    pauli_enc: tuple[str, ...],
    pauli_var: tuple[str, ...],
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
    )



@dataclass(kw_only=True)
class QuantumNativeQNetwork(SerializableModule):
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
    n_blocks_per_layer: int
    """Number of StronglyEntangled blocks per network layer."""
    rngs_kwargs: dict[str, int]
    pauli_enc: tuple[str, ...]
    """List of Pauli operator strings for encoding layer (e.g., 'X', 'Y', 'Z', 'RX', 'RY', 'RZ')."""
    pauli_var: tuple[str, ...]
    """List of Pauli operator strings for variational layer (e.g., 'X', 'Y', 'Z', 'RX', 'RY', 'RZ')."""
    device_name: str = "default.qubit"
    diff_method: str | None = None
    shots: int = 100
    """Number of circuit shots used to compute action probability distribution."""
    
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
        # self.atoms = jnp.linspace(start=self.z_min, stop=self.z_max, num=self.n_returns)
        self.atoms = jnp.array(np.linspace(start=self.z_min, stop=self.z_max, num=self.n_returns))
        # self.atoms = jnp.asarray(np.linspace(start=self.z_min, stop=self.z_max, num=self.n_returns))
        # Create non-linear trace non-increasing Kraus matrix of returns atoms.
        atoms_values = self.atoms
        atoms_values = atoms_values - atoms_values.min() # Start from zero.
        atoms_values = atoms_values / atoms_values.sum() # Normalize to probability distribution.
        # atoms_values = nnx.softmax(atoms_values) # Normalize to probability distribution.
        atoms_values = jnp.sqrt(atoms_values * self.n_returns) # Ensure that L2-norm of quantum state amplitude will produce the atom probability (i.e., $|amp|^2 = prob$).
        atoms_values = atoms_values + 1e-5 # Small offset to ensure the lowest value is not zero.
        atoms_values = atoms_values / atoms_values.max() # Ensure maximum value is 1.
        # Pad with zeros if return atoms shape is not a power of 2.
        if self.n_returns != 2**self.d_returns:
            atoms_values = jnp.pad(atoms_values, (0,2**self.d_returns - self.n_returns), mode='constant')
        # Create quantum operator from the atom values.
        self.U_atoms = jnp.diag(atoms_values)
        
        # Weights for return-atom distribution generation.
        # Every action gets their own return-atom distribution unitary.
        # self.circ_weights_var_shape = (self.n_actions,self.n_layers+1,self.d_returns,3,) # Each qubit has (Rz, Ry, Rz).
        # self.circ_weights_var_shape = (self.n_actions,self.n_layers+1,self.d_returns,2,) # Each qubit has (Rz, Ry, Rz).
        # self.circ_weights_var_shape = (self.n_actions,self.n_layers+1,self.d_returns,len(self.pauli_var),) # Each qubit has (Rz, Ry, Rz).
        self.circ_weights_var_shape = (self.n_actions,self.n_layers,self.d_returns,len(self.pauli_var),) # Each qubit has (Rz, Ry, Rz).
        # self.circ_weights_var_shape = (self.n_actions,self.n_layers,self.n_blocks_per_layer,self.d_returns,len(self.pauli_var),) # Each qubit has (Rz, Ry, Rz).
        # self.circ_weights_var_shape = (self.n_actions,self.n_layers,*qml.StronglyEntanglingLayers.shape(n_layers=self.n_blocks_per_layer, n_wires=self.d_returns),) # Each qubit has (Rz, Ry, Rz).
        ######
        # self.circ_weights_var = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_var_shape, minval=0, maxval=jnp.pi))
        ###
        self.circ_weights_var = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_var_shape, minval=0, maxval=2*jnp.pi))
        ###
        # self.circ_weights_var = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_var_shape, minval=0, maxval=1.))
        # self.circ_weights_var = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_var_shape, minval=0, maxval=jnp.pi/2.))
        # self.circ_weights_var = nnx.Param(jax.random.normal(self.rngs.params(), shape=self.circ_weights_var_shape))
        ######
        # self.circ_weights_var = jax.random.uniform(self.rngs.params(), shape=self.circ_weights_var_shape, minval=0, maxval=jnp.pi) # Not trainable.
        #
        # self.circ_weights_enc_obs_shape = (self.n_layers,self.d_returns,1,) # Encode using (Rx,).
        # self.circ_weights_enc_obs_shape = (self.n_actions,self.n_layers,self.d_returns,1,) # Encode using (Rx,).
        # self.circ_weights_enc_obs_shape = (self.d_returns,len(self.pauli_enc),) # Encode using (Rx,).
        # self.circ_weights_enc_obs_shape = (self.n_actions,self.d_returns,len(self.pauli_enc),) # Encode using (Rx,).
        self.circ_weights_enc_obs_shape = (self.n_actions,self.n_layers,self.d_returns,len(self.pauli_enc),) # Encode using (Rx,).
        # self.circ_weights_enc_obs_shape = (self.n_layers,self.d_returns,len(self.pauli_enc),) # Encode using (Rx,).
        # self.circ_weights_enc_obs = nnx.Param(jnp.ones(shape=self.circ_weights_enc_obs_shape)) # default
        # self.circ_weights_enc_obs = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_enc_obs_shape, minval=0, maxval=1.))
        ###
        self.circ_weights_enc_obs = nnx.Param(jax.random.uniform(self.rngs.params(), shape=self.circ_weights_enc_obs_shape, minval=-1., maxval=1.))
        ###
        # self.circ_weights_enc_obs = nnx.Param(jax.random.normal(self.rngs.params(), shape=self.circ_weights_enc_obs_shape))
        # self.circ_weights_enc_obs = jnp.ones(shape=self.circ_weights_enc_obs_shape) # Not trainable.
        
        
        # Inputs to circuit will be encoded onto the return-atom wires.
        self.circ_observation_shape_in = self.din # Input dimension of the observations.
        # self.circ_observation_shape_out = (self.n_actions,self.d_returns,1,) # Each qubit has (Rx,).
        # self.circ_observation_shape_out = (self.n_actions,self.n_layers,self.d_returns,1,) # Each qubit has (Rx,).
        # self.circ_observation_shape_out = (self.n_actions,self.n_layers+1,self.d_returns,1,) # Each qubit has (Rx,).
        # self.circ_observation_shape_out = (self.n_actions,self.n_layers+1,self.d_returns,len(self.pauli),) # Each qubit has (Rx,).
        # self.circ_observation_shape_out = (self.n_layers,self.d_returns,len(self.pauli_enc),) # Each qubit has (Rx,).
        self.circ_observation_shape_out = (self.d_returns,len(self.pauli_enc),)
        # self.circ_observation_shape_out = (self.n_actions,self.d_returns,len(self.pauli_enc),)
        # self.circ_observation_shape_out = (self.d_returns,len(self.pauli_enc),) # Each qubit has (Rx,).
        # self.circ_observation_shape_out = (self.n_actions,self.n_layers,self.d_returns,len(self.pauli_enc),)
        # self.layer_project_input = nnx.Linear(in_features=self.circ_observation_shape_in, out_features=reduce(lambda x,y: x*y, self.circ_observation_shape_out), rngs=self.rngs)
        
        self.linears = [
            # nnx.Linear(in_features=self.din, out_features=120, rngs=self.rngs),
            # nnx.Linear(in_features=120, out_features=84, rngs=self.rngs),
            # nnx.Linear(in_features=84, out_features=reduce(lambda x,y: x*y, self.circ_observation_shape_out), rngs=self.rngs),
            # 
            # nnx.Linear(in_features=120, out_features=reduce(lambda x,y: x*y, self.circ_observation_shape_out), rngs=self.rngs),
            # 
            nnx.Linear(in_features=self.din, out_features=reduce(lambda x,y: x*y, self.circ_observation_shape_out), rngs=self.rngs),
        ]
        
        assert self.shots is None or self.shots is not None and self.shots > 0, f"number if shots is invalid: {self.shots}"
    
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
    def project_observations(self, observations: jax.Array): # (batch,din,)
        ### Project observation into qubit space using cascade of fully-connected layers.
        x = observations
        for layer in self.linears[:-1]:
            x = layer(x)
            x = nnx.relu(x)
        x = self.linears[-1](x)
        x = jnp.reshape(x, (-1, *self.circ_observation_shape_out,))
        ### Multiplex the encoded observations across the layers of each action operator.
        # x = 2. * jnp.arctan(x) + jnp.pi # [0, 2*pi]
        x = 2. * jnp.arctan(x) # [0, pi]
        # x = nnx.relu(x)
        x = jnp.einsum("alqf,bqf->balqf", self.circ_weights_enc_obs, x)
        # x = 2. * jnp.arctan(x) + jnp.pi # [0, 2*pi]
        # x = jnp.einsum("aqf,bqf->baqf", self.circ_weights_enc_obs, x)
        # x = jnp.einsum("qf,bqf->bqf", self.circ_weights_enc_obs, x)
        # x = jnp.einsum("lqf,bqf->blqf", self.circ_weights_enc_obs, x)
        # x = x = nnx.relu(x)
        # print(f"{x.shape = }")
        # print(f"{observations.shape = }")
        # print(f"{angles_enc.shape = }")
        return x
        ###########
        # # Project input observation into shape that is compatible with the quantum circuit.
        # observation_proj_batched = self.layer_project_input(observations) # (batch * n_actions * d_returns * 1,)
        # # Reshape into batched vector with predefined shape for observation data as input to the circuit.
        # observation_proj_batched = jnp.reshape(observation_proj_batched, (-1, *self.circ_observation_shape_out,)) # (batch, n_actions, d_returns, 1,)
        # angles_enc = observation_proj_batched
        # # angles_enc = jnp.arctan(observation_proj_batched) # [-pi/2, pi/2]
        # angles_enc = 2. * jnp.arctan(angles_enc) + jnp.pi # [0, 2*pi]
        # return angles_enc
        ############
        # #
        # # Transform projected observations into encoding angles.
        # # Einsum dimension labels:
        # #   b = batch
        # #   l = layer
        # #   q = qubit
        # #   f = feature
        # # angles_enc = jnp.einsum("lqf,bqf->blqf", self.circ_weights_enc_obs, observation_proj_batched) # (batch, n_layers, d_returns, 1,)
        # # angles_enc = jnp.einsum("aqf,bqf->baqf", self.circ_weights_enc_obs, observation_proj_batched) # (batch, n_layers, d_returns, 1,)
        # angles_enc = jnp.einsum("alqf,bqf->balqf", self.circ_weights_enc_obs, observation_proj_batched) # (batch, n_layers, d_returns, 1,)
        # # angles_enc = jnp.einsum("alqf,baqf->balqf", self.circ_weights_enc_obs, observation_proj_batched) # (batch, n_actions, n_layers, d_returns, 1,)
        # ## Squash encoding angles.
        # # angles_enc = jnp.arctan(angles_enc) # [-pi/2, pi/2]
        # # angles_enc = 2. * jnp.arctan(angles_enc) # [-pi, pi]
        # angles_enc = 2. * jnp.arctan(angles_enc) + jnp.pi # [0, 2*pi]
        # #
        # return angles_enc
    
    
    # @nnx.jit
    # def call_gen_action_dist(self, x): # (batch,din,)
    @nnx.jit(static_argnames=['n_reuploads'])
    def call_gen_action_dist(self, x, n_reuploads: int | None = None): # (batch,din,)
        """Generate action distribuion from input observation."""
        
        # Track the batch size.
        batch_size = x.shape[0] if len(x.shape) > 1 else 1 if len(x.shape) > 1 else 1
        
        # Project input observation into shape that is compatible with the quantum circuit.
        observation_proj_batched = self.project_observations(x) # (batch, n_layers, d_returns, 1,)
        
        # Prepare the n-th power of the m-th moment of the returns distribution.
        key = self.rngs.default() # Generate new random key for circuit measurement.
        probs_actions = jnp.ones(shape=(batch_size, self.n_actions,), dtype='float64')
        if n_reuploads is None:
            n_reuploads = self.n_reuploads
        # for _ in range(self.n_reuploads):
        # Create a batch of keys.
        # subkeys = jax.random.split(key=key, num=batch_size)
        subkeys = [key]
        for _ in range(n_reuploads):
            # jax.debug.print("[QuantumNativeQNetworkCascade] Upload {n}: start, {probs_actions}", n=n, probs_actions=probs_actions)
            # Run quantum circuit.
            # key, subkey = jax.random.split(key=key)
            # probs_actions = self.run_circuit_action_dist( # (batch,2**d_actions,)
            #     # All device and QNode parameters.
            #     # key=subkey,
            #     key=subkeys,
            #     device_name=self.device_name,
            #     # device_name="default.mixed",
            #     # shots=self.shots,
            #     # diff_method=self.diff_method,
            #     shots=100, diff_method=None, # Shots with no diffmethod ensures that memory doesn't explore with reset operations.
            #     # Circuit arguments.
            #     inputs=x,
            #     probs_actions=probs_actions,
            #     weights=self.circ_weights,
            #     n_actions=self.n_actions,
            #     d_actions=self.d_actions,
            #     d_returns=self.d_returns,
            #     m_moment=self.m_moment,
            #     n_layers=self.n_layers,
            #     U_atoms=self.U_atoms,
            # )
            # Create a batch of keys for the current reupload.
            subkeys = jax.random.split(key=subkeys[0], num=batch_size)
            # Run the circuit on batched input.
            probs_actions = batched_run_circuit_action_dist( # (batch,2**d_actions,)
                # All device and QNode parameters.
                key=subkeys,
                # 
                # device_name=self.device_name,
                # shots=self.shots,
                # diff_method=self.diff_method,
                # 
                device_name="default.qubit",
                # device_name="default.mixed",
                # shots=self.shots,
                shots=None,
                diff_method=None,
                # 
                # shots=100, diff_method=None, # Shots with no diffmethod ensures that memory doesn't explore with reset operations.
                # shots=1000, diff_method=None, # Shots with no diffmethod ensures that memory doesn't explore with reset operations.
                # Circuit arguments.
                inputs=observation_proj_batched,
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
            )
            # jax.debug.print("a = {}", a)
            # Filter circuit output based on number of "active" actions.
            # If desired number of actions is not a multiple of 2, then filter out the excess actions on the qubit register (i.e., more actions can be represented on the register and we don't care about them).
            probs_actions = probs_actions[:,jnp.array(self.active_action_idx, dtype=int)] # (batch,n_actions,)
            # Ensure probabilities are normalized after the filter.
            # probs_actions = probs_actions / probs_actions.sum(axis=-1, keepdims=True)
            probs_actions = nnx.softmax(probs_actions, axis=-1)
            # jax.debug.print("probs_actions.shape: {}", probs_actions.shape)
            # jax.debug.print("nnx.softmax(probs_actions, axis=-1): {}", nnx.softmax(probs_actions, axis=-1))
            # jax.debug.print("nnx.softmax(probs_actions, axis=0): {}", nnx.softmax(probs_actions, axis=0))
            # jax.debug.print("probs_actions / probs_actions.sum(axis=-1, keepdims=True): {}", probs_actions / probs_actions.sum(axis=-1, keepdims=True))
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
        
        # Track the batch size.
        batch_size = x.shape[0] if len(x.shape) > 1 else 1 if len(x.shape) > 1 else 1
        
        # Project input observation into shape that is compatible with the quantum circuit.
        observation_proj_batched = self.project_observations(x) # (batch, n_layers, d_returns, 1,)
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()
        
        # Create a batch of keys.
        subkeys = jax.random.split(key=key, num=batch_size)
        
        # Run the circuit.
        probs_returns = batched_run_circuit_return_dist( # (batch,2**d_returns,)
            # All device and QNode parameters.
            key=subkeys,
            device_name=self.device_name,
            # shots=self.shots, diff_method=self.diff_method,
            shots=None, diff_method='backprop',
            # Circuit arguments.
            inputs=observation_proj_batched,
            probs_actions=probs_actions,
            weights=self.circ_weights_var,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            n_layers=self.n_layers,
            pauli_enc=self.pauli_enc,
            pauli_var=self.pauli_var,
        )
        # Filter circuit output based on number of "active" returns.
        # If desired number of returns is not a multiple of 2, then filter out the excess returns on the qubit register (i.e., more returns can be represented on the register and we don't care about them).
        probs_returns = probs_returns[:,jnp.array(self.active_return_idx, dtype=int)] # (batch,n_returns,)
        # Ensure probabilities are normalized after the filter.
        # probs_returns = probs_returns / probs_returns.sum(axis=-1, keepdims=True)
        return probs_returns
    
    @nnx.jit
    def call_gen_return_dist_NEW(self, x, probs_actions): # (batch,din,)
        """Generate return distribuion from input observation."""
        
        # Track the batch size.
        batch_size = x.shape[0] if len(x.shape) > 1 else 1 if len(x.shape) > 1 else 1
        
        # Project input observation into shape that is compatible with the quantum circuit.
        observation_proj_batched = self.project_observations(x) # (batch, n_layers, d_returns, 1,)
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()

        # Create a batch of keys.
        subkeys = jax.random.split(key=key, num=batch_size)
        
        # Run the circuit.
        probs_returns = batched_run_circuit_return_dist( # (batch,2**d_returns,)
            # All device and QNode parameters.
            key=subkeys,
            #
            # device_name=self.device_name,
            # shots=None, diff_method='backprop',
            # 
            # device_name='default.mixed', # Causes `nan` loss.
            device_name='default.qubit',
            shots=None, diff_method='backprop',
            #
            # Circuit arguments.
            inputs=observation_proj_batched,
            probs_actions=probs_actions,
            weights=self.circ_weights_var,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            n_layers=self.n_layers,
            pauli_enc=self.pauli_enc,
            pauli_var=self.pauli_var,
        )
        # jax.debug.print("probs_returns[0]: {}", probs_returns[0])
        # jax.debug.print("probs_returns[0].sum(): {}", probs_returns[0].sum())
        # jax.debug.print("nnx.softmax(probs_returns[0]): {}", nnx.softmax(probs_returns[0]))
        # jax.debug.print("nnx.softmax(probs_returns[0]).sum(): {}", nnx.softmax(probs_returns[0]).sum())
        # Filter circuit output based on number of "active" returns.
        # If desired number of returns is not a multiple of 2, then filter out the excess returns on the qubit register (i.e., more returns can be represented on the register and we don't care about them).
        probs_returns = probs_returns[:,jnp.array(self.active_return_idx, dtype=int)] # (batch,n_returns,)
        ######### DELETE THIS IF PERFORMANCE SUFFERS.
        # Ensure probabilities are normalized after the filter.
        # probs_returns = probs_returns / probs_returns.sum(axis=-1, keepdims=True)
        # probs_returns = nnx.softmax(probs_returns, axis=-1)
        ########
        # jax.debug.print("probs_returns.sum(axis=-1): {}", probs_returns.sum(axis=-1))
        # jax.debug.print("NEW probs_returns[0].sum(): {}", probs_returns[0].sum())
        # jax.debug.print("probs_returns: {}", probs_returns)
        # jax.debug.print("probs_returns / probs_returns.sum(axis=-1, keepdims=True): {}", probs_returns / probs_returns.sum(axis=-1, keepdims=True))
        # jax.debug.print("nnx.softmax(probs_returns, axis=-1): {}", nnx.softmax(probs_returns, axis=-1))
        return probs_returns
    
    @nnx.jit
    def call_gen_return_dist_shifted(self, x, probs_actions, U_shift): # (batch,din,)
        """Generate shifted return distribuion from input observation and action probability distribution."""
        
        # Track the batch size.
        batch_size = x.shape[0] if len(x.shape) > 1 else 1 if len(x.shape) > 1 else 1
        
        # Project input observation into shape that is compatible with the quantum circuit.
        observation_proj_batched = self.project_observations(x) # (batch, n_layers, d_returns, 1,)
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()

        # Create a batch of keys.
        subkeys = jax.random.split(key=key, num=batch_size)
        
        # Run the circuit.
        probs_returns = batched_run_circuit_return_dist_shifted( # (batch,2**d_returns,)
            # All device and QNode parameters.
            key=subkeys,
            #
            # device_name=self.device_name,
            # shots=None, diff_method='backprop',
            # 
            device_name='default.mixed',
            shots=None, diff_method='backprop',
            #
            # Circuit arguments.
            inputs=observation_proj_batched,
            probs_actions=probs_actions,
            U_shift=U_shift,
            weights=self.circ_weights_var,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            n_layers=self.n_layers,
            pauli_enc=self.pauli_enc,
            pauli_var=self.pauli_var,
        )
        # Filter circuit output based on number of "active" returns.
        # If desired number of returns is not a multiple of 2, then filter out the excess returns on the qubit register (i.e., more returns can be represented on the register and we don't care about them).
        probs_returns = probs_returns[:,jnp.array(self.active_return_idx, dtype=int)] # (batch,n_returns,)
        # Ensure probabilities are normalized after the filter.
        probs_returns = probs_returns / probs_returns.sum(axis=-1, keepdims=True)
        # probs_returns = nnx.softmax(probs_returns, axis=-1)
        return probs_returns
    
    
    @nnx.jit
    def call_hermitian_loss(self, x, probs_actions, rho): # (batch,din,)
        """Compute Hermitian loss across the return-atom qubits with respect to an input density matrix rho."""
        
        # Track the batch size.
        batch_size = x.shape[0] if len(x.shape) > 1 else 1 if len(x.shape) > 1 else 1
        
        # Project input observation into shape that is compatible with the quantum circuit.
        observation_proj_batched = self.project_observations(x) # (batch, n_layers, d_returns, 1,)
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()

        # Create a batch of keys.
        subkeys = jax.random.split(key=key, num=batch_size)
        
        # Run the circuit.
        loss = batched_run_circuit_hermitian_loss( # (batch,2**d_returns,)
            # All device and QNode parameters.
            key=subkeys,
            # device_name=self.device_name,
            # device_name="default.mixed", # Use mixed state simulation for density matrix.
            device_name="default.qubit",
            # shots=self.shots, diff_method=self.diff_method,
            shots=None, diff_method='backprop',
            # Circuit arguments.
            inputs=observation_proj_batched,
            probs_actions=probs_actions,
            weights=self.circ_weights_var,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            n_layers=self.n_layers,
            rho=rho,
            pauli_enc=self.pauli_enc,
            pauli_var=self.pauli_var,
        )
        return loss
    
    @nnx.jit
    def call_return_density_matrix(self, x, probs_actions): # (batch,din,)
        """Generate the density matrix across the return-atom qubits according to an input observation and action probability distribution."""
        
        # Track the batch size.
        batch_size = x.shape[0] if len(x.shape) > 1 else 1 if len(x.shape) > 1 else 1
        
        # Project input observation into shape that is compatible with the quantum circuit.
        observation_proj_batched = self.project_observations(x) # (batch, n_layers, d_returns, 1,)
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()

        # Create a batch of keys.
        subkeys = jax.random.split(key=key, num=batch_size)
        
        # Run the circuit.
        rho = batched_run_circuit_return_density_matrix( # (batch,2**d_returns,2**d_returns)
            # All device and QNode parameters.
            key=subkeys,
            # device_name=self.device_name,
            # device_name="default.mixed", # Use mixed state simulation for density matrix.
            device_name="default.qubit", # Use mixed state simulation for density matrix.
            # shots=self.shots, diff_method=self.diff_method,
            shots=None, diff_method='backprop',
            # Circuit arguments.
            inputs=observation_proj_batched,
            probs_actions=probs_actions,
            weights=self.circ_weights_var,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            n_layers=self.n_layers,
            pauli_enc=self.pauli_enc,
            pauli_var=self.pauli_var,
        )
        return rho
    
    @nnx.jit
    def call_return_density_matrix_shifted(self, x, probs_actions, U_shift): # (batch,din,)
        """Generate the shifted density matrix across the return-atom qubits according to an input observation and action probability distribution."""
        
        # Track the batch size.
        batch_size = x.shape[0] if len(x.shape) > 1 else 1 if len(x.shape) > 1 else 1
        
        # Project input observation into shape that is compatible with the quantum circuit.
        observation_proj_batched = self.project_observations(x) # (batch, n_layers, d_returns, 1,)
        
        # Generate new random key for circuit measurement.
        key = self.rngs.default()

        # Create a batch of keys.
        subkeys = jax.random.split(key=key, num=batch_size)
        
        # Run the circuit.
        rho = batched_run_circuit_return_density_matrix_shifted( # (batch,2**d_returns,2**d_returns)
            # All device and QNode parameters.
            key=subkeys,
            # device_name=self.device_name,
            # device_name="default.mixed", # Use mixed state simulation for density matrix.
            device_name="default.qubit", # Use mixed state simulation for density matrix.
            # shots=self.shots, diff_method=self.diff_method,
            shots=None, diff_method='backprop',
            # Circuit arguments.
            inputs=observation_proj_batched,
            probs_actions=probs_actions,
            U_shift=U_shift,
            weights=self.circ_weights_var,
            n_actions=self.n_actions,
            d_actions=self.d_actions,
            d_returns=self.d_returns,
            n_layers=self.n_layers,
            pauli_enc=self.pauli_enc,
            pauli_var=self.pauli_var,
        )
        return rho