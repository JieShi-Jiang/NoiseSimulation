"""
Estimation hooks for large-n analysis (probabilities, small-n density matrices).

- ``yk_noiseless_binary``: exact P(last = |0>) under noiseless simulation (noise stripped).
- ``simulate_density_matrix`` / ``prob_last_qubit_zero_dm``: small-n noisy rho paths.
- ``probabilities_last_readout_basis_state_tn``: large-n noisy expectation values
  ``<x| E^\dagger(M_k) |x>`` for computational-basis product inputs without
  forming dense ``rho``.
- ``strip_noise_gates``: unitary-only surrogate circuit.
"""

from __future__ import annotations

import os
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import cirq
import numpy as np
import tensornetwork as tn

from qlipschitz import circuit_to_tensor


def strip_noise_gates(circuit: cirq.Circuit) -> cirq.Circuit:
    """Remove channel operations; keep unitary gates (surrogate for probability probes)."""
    out = cirq.Circuit()
    for op in circuit.all_operations():
        if cirq.has_unitary(op):
            out.append(op)
    return out


def _embed_single_qubit_operator(
    sorted_qubits: Sequence[cirq.GridQubit], target: cirq.GridQubit, mat2: np.ndarray
) -> np.ndarray:
    """Kronecker-expand a single-qubit 2x2 onto ``sorted_qubits`` ordering."""
    mat2 = np.asarray(mat2, dtype=np.complex128)
    mats = [np.eye(2, dtype=np.complex128) for _ in sorted_qubits]
    idx = sorted_qubits.index(target)
    mats[idx] = mat2
    op = mats[0]
    for m in mats[1:]:
        op = np.kron(op, m)
    return op


def _embed_joint_last_two_operator(
    sorted_qubits: Sequence[cirq.GridQubit],
    mat4: np.ndarray,
) -> np.ndarray:
    """Kronecker-expand a 4x4 operator onto the last two qubits."""
    if len(sorted_qubits) < 2:
        raise ValueError("joint last-two operator requires at least two qubits")
    mat4 = np.asarray(mat4, dtype=np.complex128)
    if mat4.shape != (4, 4):
        raise ValueError(f"Expected a (4, 4) operator, got {mat4.shape}")

    prefix = np.eye(1, dtype=np.complex128)
    for _ in range(len(sorted_qubits) - 2):
        prefix = np.kron(prefix, np.eye(2, dtype=np.complex128))
    return np.kron(prefix, mat4)


def qubit_projector(bit: int) -> np.ndarray:
    if bit == 0:
        return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    if bit == 1:
        return np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    raise ValueError(f"bit must be 0 or 1, got {bit}")


def projector_comp_basis(k: int) -> np.ndarray:
    """Rank-1 projector |k><k| on the last two qubits, k in {0,1,2,3}."""
    if not 0 <= int(k) <= 3:
        raise ValueError(f"k must be in {{0,1,2,3}}, got {k}")
    proj = np.zeros((4, 4), dtype=np.complex128)
    proj[int(k), int(k)] = 1.0
    return proj


def yk_noiseless_binary_last_qubit(circuit: cirq.Circuit, qubits: Sequence[cirq.GridQubit]) -> float:
    """P(measure 0) on ``target`` last GridQubit under noiseless unitary simulation."""
    Ucirc = strip_noise_gates(circuit)
    sim = cirq.Simulator()
    qubit_order = sorted(Ucirc.all_qubits())
    state = sim.simulate(Ucirc, qubit_order=qubit_order).final_state_vector
    last = sorted(qubits)[-1]
    idx_last = qubit_order.index(last)
    mask_for_last_0 = (np.arange(len(state)) >> idx_last) & 1 == 0
    return float(np.sum(np.abs(state[mask_for_last_0]) ** 2))


def probabilities_last_readout_noiseless(
    circuit: cirq.Circuit,
    qubits: Sequence[cirq.GridQubit],
    *,
    n_classes: int,
) -> np.ndarray:
    """Noiseless readout probabilities on the last 1 or 2 qubits."""
    sorted_qubits = sorted(qubits)
    Ucirc = strip_noise_gates(circuit)
    sim = cirq.Simulator()
    qubit_order = sorted(Ucirc.all_qubits())
    state = np.asarray(sim.simulate(Ucirc, qubit_order=qubit_order).final_state_vector).reshape(-1)

    if int(n_classes) == 2:
        last = sorted_qubits[-1]
        idx_last = qubit_order.index(last)
        mask0 = ((np.arange(len(state)) >> idx_last) & 1) == 0
        p0 = float(np.sum(np.abs(state[mask0]) ** 2))
        return np.array([p0, max(0.0, 1.0 - p0)], dtype=np.float64)

    if int(n_classes) == 4:
        qa, qb = sorted_qubits[-2], sorted_qubits[-1]
        idx_a = qubit_order.index(qa)
        idx_b = qubit_order.index(qb)
        probs = np.zeros((4,), dtype=np.float64)
        for basis_index, amp in enumerate(state):
            bit_a = (basis_index >> idx_a) & 1
            bit_b = (basis_index >> idx_b) & 1
            probs[(bit_a << 1) | bit_b] += float(np.abs(amp) ** 2)
        return probs

    raise ValueError(f"Only 2-class and 4-class readout are supported, got n_classes={n_classes}")


def simulate_density_matrix(circuit: cirq.Circuit) -> np.ndarray:
    """Return rho (only feasible for small n)."""
    sim = cirq.DensityMatrixSimulator()
    return np.asarray(sim.simulate(circuit).final_density_matrix)


def prob_last_qubit_zero_dm(rho: np.ndarray, sorted_qubits: Sequence[cirq.GridQubit]) -> float:
    """Tr(|0><0| ⊗ I_rest  rho) via (1 + <Z_last>)/2 on embedded Pauli."""
    last = sorted_qubits[-1]
    z_full = _embed_single_qubit_operator(sorted_qubits, last, np.array([[1, 0], [0, -1]], dtype=np.complex128))
    z_exp = np.trace(z_full @ rho)
    return float(np.real(0.5 * (1.0 + z_exp)))


def yk_exact_dm_binary_last(circuit: cirq.Circuit, qubits: Sequence[cirq.GridQubit]) -> float:
    """Noisy Born probability P(last = |0>) via density-matrix simulator."""
    rho = simulate_density_matrix(circuit)
    return prob_last_qubit_zero_dm(rho, sorted(qubits))


def probabilities_last_readout_exact_dm(
    circuit: cirq.Circuit,
    qubits: Sequence[cirq.GridQubit],
    *,
    n_classes: int,
) -> np.ndarray:
    """Noisy Born probabilities on the last 1 or 2 qubits via density matrix."""
    rho = simulate_density_matrix(circuit)
    sorted_qubits = sorted(qubits)

    if int(n_classes) == 2:
        p0 = prob_last_qubit_zero_dm(rho, sorted_qubits)
        return np.array([p0, max(0.0, 1.0 - p0)], dtype=np.float64)

    if int(n_classes) == 4:
        probs = np.zeros((4,), dtype=np.float64)
        for k in range(4):
            op = _embed_joint_last_two_operator(sorted_qubits, projector_comp_basis(k))
            probs[k] = float(np.real(np.trace(op @ rho)))
        return probs

    raise ValueError(f"Only 2-class and 4-class readout are supported, got n_classes={n_classes}")


def _connect_basis_vector(nodes_set: list[tn.Node], edge: tn.Edge, bit: int) -> None:
    vec = np.array([1.0, 0.0], dtype=np.complex128) if int(bit) == 0 else np.array([0.0, 1.0], dtype=np.complex128)
    node = tn.Node(vec, axis_names=[edge.name])
    nodes_set.append(node)
    edge ^ node[edge.name]


def expectation_basis_state_effective_observable(
    circuit: cirq.Circuit,
    qubits: Sequence[cirq.GridQubit],
    *,
    measurement: np.ndarray | None = None,
    joint_measurement: np.ndarray | None = None,
    basis_bits: Sequence[int] | None = None,
) -> float:
    """
    Compute ``<b| E^dagger(M) |b>`` with tensor contraction for a computational basis product state.

    This keeps memory O(n) in the input description and avoids building dense ``rho``
    when the input is a basis product state such as ``|0...0>``.
    """
    sorted_qubits = list(sorted(qubits))
    if basis_bits is None:
        basis_bits = [0] * len(sorted_qubits)
    basis_bits = [int(bit) for bit in basis_bits]
    if len(basis_bits) != len(sorted_qubits):
        raise ValueError(
            f"basis_bits length {len(basis_bits)} does not match qubit count {len(sorted_qubits)}"
        )
    if measurement is None:
        measurement = np.zeros((2, 2), dtype=np.complex128)

    nodes_set, left_edge, right_edge = circuit_to_tensor(
        circuit,
        sorted_qubits,
        np.asarray(measurement, dtype=np.complex128),
        joint_measurement=None if joint_measurement is None else np.asarray(joint_measurement, dtype=np.complex128),
    )

    for bit, edge in zip(basis_bits, left_edge):
        _connect_basis_vector(nodes_set, edge, bit)
    for bit, edge in zip(basis_bits, right_edge):
        _connect_basis_vector(nodes_set, edge, bit)

    scalar = tn.contractors.auto(nodes_set).tensor
    return float(np.real(np.asarray(scalar)))


def probabilities_last_readout_basis_state_tn(
    circuit: cirq.Circuit,
    qubits: Sequence[cirq.GridQubit],
    *,
    n_classes: int,
    basis_bits: Sequence[int] | None = None,
) -> np.ndarray:
    """Large-n noisy readout probabilities ``<b|E^dagger(P_k)|b>`` from TN contraction."""
    sorted_qubits = sorted(qubits)
    if basis_bits is None:
        basis_bits = [0] * len(sorted_qubits)

    if int(n_classes) == 2:
        probs = np.zeros((2,), dtype=np.float64)
        for bit in (0, 1):
            probs[bit] = expectation_basis_state_effective_observable(
                circuit,
                sorted_qubits,
                measurement=qubit_projector(bit),
                joint_measurement=None,
                basis_bits=basis_bits,
            )
        return probs

    if int(n_classes) == 4:
        probs = np.zeros((4,), dtype=np.float64)
        for k in range(4):
            probs[k] = expectation_basis_state_effective_observable(
                circuit,
                sorted_qubits,
                joint_measurement=projector_comp_basis(k),
                basis_bits=basis_bits,
            )
        return probs

    raise ValueError(f"Only 2-class and 4-class readout are supported, got n_classes={n_classes}")
