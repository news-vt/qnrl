# Mathematical proof for amplitude kickback

Quantum system with two hilbert spaces, one for actions $a$ and the other for retuns $z$, denoted by $\mathcal{H}_{a} \otimes \mathcal{H}_{z}$.

We prepare an initial state of superposition for the actions via:

$$
\ket{\psi_0} = \sum_{a} \sqrt{p(a)} \ket{a} \otimes \ket{0}^{\otimes |Z|}
$$

Define an operator over the combined Hilbert space $\mathcal{H}_{a} \otimes \mathcal{H}_{z}$ for preparing returns probability distribution for every action -- this would be the ML model -- called $U_z$

$$
\begin{aligned}
U_{z|a}
    &= \sum_{a} \ket{a}\bra{a} \otimes U_{z}(a) \\
    &= \sum_{a} \ket{a}\bra{a} \otimes \Bigl[ \sum_{z} \sqrt{p(z|a)} \ket{z}\bra{0} \Bigr],
\end{aligned}
$$

which creates the state:
$$
U_{z|a} \ket{\psi_0} = \sum_{a} \sqrt{p(a)} \ket{a} \otimes \Bigl[ \sum_{z} \sqrt{p(z|a)} \ket{z} \Bigr].
$$

Now define an operator over the return-atoms Hilbert space $\mathcal{H}_{z}$ that encodes the return-atom values called $U_{v|z}$

$$
\begin{aligned}
U_{v|z} 
    &= \sum_{z} \sqrt{v_z} \ket{z}\bra{z}
    &= \begin{bmatrix}
    \sqrt{v_0} & 0 & 0 & \dots \\
    0 & \sqrt{v_1} & 0 &  \\
    0 & 0 & \sqrt{v_2} &  \\
    \vdots &  &  & \ddots
    \end{bmatrix}
\end{aligned},
$$

which is a diagonal matrix, where $\sum_{z} v_z = 1$, i.e., the return-atom values form a probability distribution. Because the values form a probability distribution, we also have that $U_{v|z}$ is completely positive and $U_{v|z} \leq \mathbb{I}$, which means that $U_{v|z}$ is trace non-increasing. Notice that $U_{v|z}$ is Hermitian (i.e., eigenvalues are real), but not unitary. This means that the resulting state after applying $U_{v|z}$ will not be normalized. This normalization, however, can be circumvented by applying a reset operation on the return-atom space, as we will see shortly.

Applying both operators in succession on $\ket{\psi_0}$ produces the state

$$
\begin{aligned}
(\mathbb{I} \otimes U_{v|z})  U_{z|a} \ket{\psi_0} 
    &= \sum_{a} \sqrt{p(a)} \ket{a} \otimes \Bigl[ \sum_{z} \sqrt{p(z|a)} \sqrt{v_z} \ket{z} \Bigr] \\
    &= \ket{\phi_0},
\end{aligned}
$$

which we will refer to as $\ket{\phi_0}$, where the subscript denotes the number of times the operations have been applied (this will be important later).

Now, to fix the normalization on the return-atoms space, we apply a reset operation over the space of the return-atoms $\mathcal{H}_{z}$ to collapse their qubits. A reset operation projects the selection fo qubits to the $\ket{0}$ state. To accomplish this we apply a mid-circuit measurement on the return-atom qubits ($\mathcal{H}_{z}$) to collapse their state to to one of $\ket{z} \in \{\ket{z}\}_{z\in [1,|Z|]}$ and then apply a conditional correction to map $\ket{z} \to \ket{0}$ (e.g., a bit-flip operation if $z \in \{0,1\}$). Mathematically, we can represent this collapse as a measurement projection followed by a partial-trace over the $\mathcal{H}_{z}$ space.

Before we do this, we first need to define the measurement process. A quantum measurement $M$ maps a quantum state in density matrix form $\rho$ to a mixed state using a set of projection operators $\Pi$ outlined by the measurement basis, given by

$$
M[\rho] = \sum_{i} \Pi_i \rho \Pi_i
$$

where $\rho$ is the density matrix for a quantum system of $q$ qubits, and, if measuring in the Pauli-$Z$ basis, $\Pi_i = \ket{i}\bra{i}$, $\forall i \in \{0,2^{q}-1\}$.

<!-- See -->
<!-- https://pennylane.ai/qml/demos/tutorial_mcm_introduction#qubit-reset -->