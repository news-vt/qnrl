# Mathematical proof for amplitude kickback

Quantum system with two hilbert spaces, one for actions $a$ and the other for retuns $z$, denoted by $\mathcal{H}_{a} \otimes \mathcal{H}_{z} = \mathbb{C}^{2^{|\mathcal{A}|}} \otimes \mathbb{C}^{2^{|\mathcal{Z}|}}$ where $a \in \mathcal{A}$ is the set of actions and $z \in \mathcal{Z}$ is the set of return atoms.

We start with an initial state of all zeros across the two spaces, which looks like 

```math
\ket{\psi_{\textrm{in}}} = \ket{0}^{\otimes |\mathcal{A}|} \otimes \ket{0}^{\otimes |\mathcal{Z}|}
```

We prepare an initial state of superposition for the actions via:

```math
\ket{\psi_0} = \sum_{a} \sqrt{p(a)} \ket{a} \otimes \ket{0}^{\otimes |\mathcal{Z}|}
```

It will become convenient to represent this state in density matrix form, for reasons we will see shortly. This state can be represented as the density matrix:

```math
\begin{aligned}
\rho_0 
&= \ket{\psi_0}\bra{\psi_0} \\
&= (\sum_{a_i} \sqrt{p(a_i)} \ket{a_i} \otimes \ket{0}^{\otimes |\mathcal{Z}|}) (\sum_{a_j} \sqrt{p(a_j)} \bra{a_j} \otimes \bra{0}^{\otimes |\mathcal{Z}|}) \\
&= \sum_{a_i} \sum_{a_j} \sqrt{p(a_i)} \sqrt{p(a_j)} \ket{a_i} \otimes \ket{0}^{\otimes |\mathcal{Z}|} \bra{a_i} \otimes \bra{0}^{\otimes |\mathcal{Z}|}.
\end{aligned}
```

Next, we define an operator over the combined Hilbert space $\mathcal{H}_{a} \otimes \mathcal{H}_{z}$ for preparing a probability distribution of return-atoms for every action -- this would be the ML model -- called $U_z$

```math
\begin{aligned}
U_{z|a}
    &= \sum_{a} \ket{a}\bra{a} \otimes U_{z}(a) \\
    &= \sum_{a} \ket{a}\bra{a} \otimes \Bigl[ \sum_{z} \sqrt{p(z|a)} \ket{z}\bra{0} \Bigr],
\end{aligned}
```

which, when operating on the state-vector form of the system state, behaves as follows

```math
U_{z|a} \ket{\psi_0} = \sum_{a} \sqrt{p(a)} \ket{a} \otimes \Bigl[ \sum_{z} \sqrt{p(z|a)} \ket{z} \Bigr].
```

and in density matrix form

```math
\begin{aligned}
U_{z|a} \rho_0 U_{z|a}^{\dagger}
&= U_{z|a} \ket{\psi_0} \bra{\psi_0} U_{z|a}^{\dagger} \\
&= \Bigl( \sum_{a_i} \sqrt{p(a_i)} \ket{a_i} \otimes \Bigl[ \sum_{z_j} \sqrt{p(z_j|a_i)} \ket{z_j} \Bigr] \Bigr) \Bigl( \sum_{a_k} \sqrt{p(a_k)} \bra{a_k} \otimes \Bigl[ \sum_{z_l} \sqrt{p(z_l|a_k)} \bra{z_l} \Bigr] \Bigr) \\
&= \sum_{a_i} \sum_{z_j} \sum_{a_k} \sum_{z_l} \Bigl( \sqrt{p(a_i)} \sqrt{p(z_j|a_i)} \sqrt{p(a_k)} \sqrt{p(z_l|a_k)} \Bigr) \ket{a_i} \ket{z_j} \bra{a_k} \bra{z_l}.
\end{aligned}
```

Now define an operator over the return-atoms Hilbert space $\mathcal{H}_{z}$ that encodes the return-atom values called $U_{v|z}$

```math
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
```

which is a diagonal matrix, where $\sum_{z} v_z = 1$, i.e., the return-atom values form a probability distribution. Because the values form a probability distribution, we also have that $U_{v|z}$ is completely positive and $U_{v|z} \leq \mathbb{I}$, which means that $U_{v|z}$ is trace non-increasing. Notice that $U_{v|z}$ is Hermitian (i.e., eigenvalues are real), but not unitary. This means that the resulting state after applying $U_{v|z}$ will not be normalized. This normalization, however, can be circumvented by applying a reset operation on the return-atom space, as we will see shortly.

Applying both operators in succession on $\ket{\psi_0}$ produces the state

```math
\begin{aligned}
(\mathbb{I} \otimes U_{v|z})  U_{z|a} \ket{\psi_0} 
    &= \sum_{a} \sqrt{p(a)} \ket{a} \otimes \Bigl[ \sum_{z} \sqrt{p(z|a)} \sqrt{v_z} \ket{z} \Bigr]
\end{aligned}
```

and again in density matrix form

```math
\begin{aligned}
(\mathbb{I} \otimes U_{v|z})  U_{z|a} \rho_0 U_{z|a}^{\dagger} (\mathbb{I} \otimes U_{v|z}^{\dagger})
&= (\mathbb{I} \otimes U_{v|z})  U_{z|a} \ket{\psi_0} \bra{\psi_0} U_{z|a}^{\dagger} (\mathbb{I} \otimes U_{v|z}^{\dagger}) \\
&= \Bigl( \sum_{a_i} \sqrt{p(a_i)} \ket{a_i} \otimes \Bigl[ \sum_{z_j} \sqrt{p(z_j|a_i)} \sqrt{v_{z_j}} \ket{z_j} \Bigr] \Bigr) \Bigl( \sum_{a_k} \sqrt{p(a_k)} \bra{a_k} \otimes \Bigl[ \sum_{z_l} \sqrt{p(z_l|a_k)} \sqrt{v_{z_l}} \bra{z_l} \Bigr] \Bigr) \\
&= \sum_{a_i} \sum_{z_j} \sum_{a_k} \sum_{z_l} \Bigl( \sqrt{p(a_i)} \sqrt{p(z_j|a_i)} \sqrt{v_{z_j}} \sqrt{p(a_k)} \sqrt{p(z_l|a_k)} \sqrt{v_{z_l}} \Bigr) \ket{a_i} \ket{z_j} \bra{a_k} \bra{z_l} \\
&= \rho_1 .
\end{aligned}
```
which we will refer to as $\rho_1$, where the subscript denotes the number of times the operations have been applied -- this subscript index will be important later.

Now, to fix the normalization on the return-atoms space, we apply a reset operation over the space of the return-atoms $\mathcal{H}_{z}$ to collapse their qubits. A reset operation projects the selection fo qubits to the $\ket{0}$ state. To accomplish this we apply a mid-circuit measurement on the return-atom qubits ($\mathcal{H}_{z}$) to collapse their state to to one of $\ket{z} \in \{\ket{z}\}_{z\in [1,|\mathcal{Z}|]}$ and then apply a conditional correction to map $\ket{z} \to \ket{0}$ (e.g., a bit-flip operation if $z \in \{0,1\}$). Mathematically, we can represent this collapse as a measurement projection followed by a partial-trace over the $\mathcal{H}_{z}$ space.

Before we do this, we first need to define the measurement process. A quantum measurement $M$ maps a quantum state in density matrix form $\rho$ to a mixed state using a set of projection operators $\Pi$ outlined by the measurement basis, given by

```math
\rho' = M[\rho] = \sum_{i} \Pi_i \rho \Pi^{\dagger}_i
```

where $\rho$ is the density matrix for a quantum system of $q$ qubits, and, if measuring in the Pauli-$Z$ basis, $\Pi_i = \ket{i}\bra{i}$, $\forall i \in \{0,2^{q}-1\}$. This resulting density matrix after measurement must also be normalized, which we can do by dividing by its trace (which is just the sum of its diagonal elements) as given by

```math
\hat{\rho}' = \frac{\rho'}{\textrm{tr}[\rho']}.
```

Since $\hat{\rho}'$ spans the space of both actions and return-atoms $\mathcal{H}_{a} \otimes \mathcal{H}_{z}$, to complete the reset over the return-atoms space after the projective measurement we must trace-out the return-atoms qubits and then simply set their state to zero. To do this, we define the completed reset operation

```math
\begin{aligned}
\textrm{Reset}_z [\rho] 
    &= \textrm{tr}_{z}{\bigl[ \hat{\rho}' \bigr]} \otimes \ket{0}^{\otimes |\mathcal{Z}|}\bra{0}^{\otimes |\mathcal{Z}|} \\
    &= \textrm{tr}_{z}{\Bigl[ \frac{M[\rho]}{\textrm{tr}\bigl[M[\rho]\bigr]} \Bigr]} \otimes \ket{0}^{\otimes |\mathcal{Z}|}\bra{0}^{\otimes |\mathcal{Z}|}
\end{aligned}
```

where $\textrm{tr}_{z} : \mathcal{H}_{a} \otimes \mathcal{H}_{z} \to \mathcal{H}_{z}$ is the partial trace operator which traces out (i.e., removes) the subspace $\mathcal{H}_{z}$.

So, applying this to our distributional RL setting, we can resume from $\rho_1$ and reset the return-atom qubits in the following manner. First, we define the set of projectors $\Pi = \{ \mathbb{I}\otimes\ket{z}\bra{z} \}_{z\in Z}$, which covers all possible return-atom outcomes while doing nothing to the action space, and then apply these projectors as a *partial* measurement on $\rho_1$ as follows

```math
\begin{aligned}
\rho'_1 = M[\rho_1]
    &= \sum_{z} \Pi_z \rho_1 \Pi^{\dagger}_z \\
    &= \sum_{z} \bigl(\mathbb{I}\otimes\ket{z}\bra{z}\bigr) \rho_1 \bigl(\mathbb{I}\otimes\ket{z}\bra{z}\bigr) \\
    &= \sum_{z} \sum_{a_i} \sum_{a_k} \Bigl( \sqrt{p(a_i)} \sqrt{p(z|a_i)} \sqrt{v_{z}} \sqrt{p(a_k)} \sqrt{p(z|a_k)} \sqrt{v_{z}} \Bigr) \ket{a_i} \ket{z} \bra{a_k} \bra{z} ~~\textrm{(Collapses where $z_j=z_l=z$)}
\end{aligned}
```

which has the normalization factor

```math
\begin{aligned}
\textrm{tr}\bigl[\rho'_1\bigr]
    &= \sum_{z} \sum_{a} \sqrt{p(a)} \sqrt{p(z|a)} \sqrt{v_{z}} \sqrt{p(a)} \sqrt{p(z|a)} \sqrt{v_{z}} ~~\textrm{(Sum along diagonal collapses where $a_i=a_k=a$)} \\
    &= \sum_{z} \sum_{a} p(a) p(z|a) v_{z} \\
    &= \sum_{a} p(a) \sum_{z} p(z|a) v_{z} \\
    &= \sum_{a} p(a) \mathbb{E}[Z | a] ~~\textrm{(By definition of expectation of a random variable)}.
\end{aligned}
```

thus the entire normalized density matrix is

```math
\hat{\rho}'_1
= \frac{\rho'_1}{\textrm{tr}\bigl[\rho'_1\bigr]}
= (\frac{1}{\sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot \sum_{z} \sum_{a_i} \sum_{a_k} \Bigl( \sqrt{p(a_i)} \sqrt{p(z|a_i)} \sqrt{v_{z}} \sqrt{p(a_k)} \sqrt{p(z|a_k)} \sqrt{v_{z}} \Bigr) \ket{a_i} \ket{z} \bra{a_k} \bra{z}
```

Notice that here is where the expectation of the random return appears. Next, we can find the reduced state of the action space after the partial measurement by taking the partial trace over the return-atoms space

```math
\begin{aligned}
\textrm{tr}_{z}{\bigl[ \hat{\rho}'_1 \bigr]}
&= \sum_{z} (\mathbb{I}\otimes\bra{z}) \hat{\rho}'_1 (\mathbb{I}\otimes\ket{z}) \\
&= (\frac{1}{\sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot \sum_{z} \sum_{a_i} \sum_{a_k} \Bigl( \sqrt{p(a_i)} \sqrt{p(z|a_i)} \sqrt{v_{z}} \sqrt{p(a_k)} \sqrt{p(z|a_k)} \sqrt{v_{z}} \Bigr) \ket{a_i}\bra{a_k} .
\end{aligned}
```

The final system state after applying the reset operation is therefore

```math
\begin{aligned}
\textrm{Reset}_z [\rho_1] 
    &= \textrm{tr}_{z}{\bigl[ \hat{\rho}'_1 \bigr]} \otimes \ket{0}^{\otimes |\mathcal{Z}|}\bra{0}^{\otimes |\mathcal{Z}|} \\
    &= (\frac{1}{\sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot \sum_{z} \sum_{a_i} \sum_{a_k} \Bigl( \sqrt{p(a_i)} \sqrt{p(z|a_i)} \sqrt{v_{z}} \sqrt{p(a_k)} \sqrt{p(z|a_k)} \sqrt{v_{z}} \Bigr) \ket{a_i}\bra{a_k} \otimes \ket{0}^{\otimes |\mathcal{Z}|}\bra{0}^{\otimes |\mathcal{Z}|}.
\end{aligned}
```

Notice that at this point the return-atoms space has been reduced to the all-zero state $\ket{0}^{\otimes |\mathcal{Z}|}$, which has a very similar form to $\rho_0$. This means that, after applying the reset, the system state is placed back into a form after applying superposition of actions, and is thus receptive to the sequential return-atom operations $(\mathbb{I} \otimes U_{v|z}) U_{z|a}$.

If at this point we were to stop and measure the action qubits using projectors $\Pi = \{\ket{a}\bra{a}\}_{a \in A}$ -- ignoring the return-atom space because it is already zero -- we would get the following state in density matrix form

```math
\begin{aligned}
M[\textrm{tr}_{z}{\bigl[ \hat{\rho}'_1 \bigr]}]
&= \sum_{a} \Pi_a \textrm{tr}_{z}{\bigl[ \hat{\rho}'_1 \bigr]} \Pi^{\dagger}_a \\
&= \sum_{a} \ket{a}\bra{a} \textrm{tr}_{z}{\bigl[ \hat{\rho}'_1 \bigr]} \ket{a}\bra{a} \\
&= \sum_{a} \ket{a}\bra{a} \Bigl((\frac{1}{\sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot \sum_{z} \sum_{a_i} \sum_{a_k} \bigl( \sqrt{p(a_i)} \sqrt{p(z|a_i)} \sqrt{v_{z}} \sqrt{p(a_k)} \sqrt{p(z|a_k)} \sqrt{v_{z}} \bigr) \ket{a_i}\bra{a_k} \Bigr) \ket{a}\bra{a} \\
&= (\frac{1}{\sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot \sum_{a} \sum_{z} \bigl( \sqrt{p(a_i=a)} \sqrt{p(z|a_i=a)} \sqrt{v_{z}} \sqrt{p(a_k=a)} \sqrt{p(z|a_k=a)} \sqrt{v_{z}} \bigr) \ket{a}\bra{a} ~~\textrm{(Sums collapse when $a_i = a_k = a$)} \\
&= (\frac{1}{\sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot \sum_{a} \sum_{z} p(a) p(z|a) v_{z} \ket{a}\bra{a} \\
&= (\frac{1}{\sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot \sum_{a} p(a) \sum_{z} p(z|a) v_{z} \ket{a}\bra{a} \\
&= (\frac{1}{\sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot \sum_{a} p(a) \mathbb{E}[Z | a] \ket{a}\bra{a} ~~\textrm{(By definition of expectation of a random variable)},
\end{aligned}
```

which is diagonal with normalization factor

```math
\begin{aligned}
\textrm{tr}\Bigl[ M[\textrm{tr}_{z}{\bigl[ \hat{\rho}'_1 \bigr]}] \Bigr] = (\frac{1}{\sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot \sum_{a} p(a) \mathbb{E}[Z | a] ~~\textrm{(Sum of diagonal elements)},
\end{aligned}
```

producing the normalized state

```math
\begin{aligned}
\frac{M[\textrm{tr}_{z}{\bigl[ \hat{\rho}'_1 \bigr]}]}{\textrm{tr}\Bigl[ M[\textrm{tr}_{z}{\bigl[ \hat{\rho}'_1 \bigr]}] \Bigr]}
&= (\frac{1}{(\frac{1}{\sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot \sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot (\frac{1}{\sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot \sum_{a} p(a) \mathbb{E}[Z | a] \ket{a}\bra{a} \\
&= (\frac{1}{\sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot \sum_{a} p(a) \mathbb{E}[Z | a] \ket{a}\bra{a}.
\end{aligned}
```

Notice that because the state was normalized when creating $\hat{\rho}'_1$ this state when measuring $a$ was already normalized, but for completeness we complete the process again here.

This means that the probability of measuring an arbitrary action $a$ is

```math
P[\ket{a}] = (\frac{1}{\sum_{a} p(a) \mathbb{E}[Z | a]}) \cdot p(a) \mathbb{E}[Z | a]
```

which is the **normalized weighted expectation of return-atoms for the given action $a$ relative to all other actions $A \setminus \{a\}$**.


<!-- See -->
<!-- https://pennylane.ai/qml/demos/tutorial_mcm_introduction#qubit-reset -->
<!-- https://quantumcomputing.stackexchange.com/questions/4834/density-matrix-after-measurement-on-density-matrix -->
<!-- https://physics.stackexchange.com/questions/681556/trying-to-understand-post-measurement-density-matrices-in-a-state-that-spans-2-h -->