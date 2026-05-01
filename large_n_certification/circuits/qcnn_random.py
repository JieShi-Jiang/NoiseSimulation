"""
QCNN-style and lightweight random circuits (aligned with ``evaluate_qcnn_model.py``).

Use ``create_model_circuit`` for Veri-Q QCNN geometry + optional depolarizing/bit-flip/phase-flip noise.
Use ``random_heisenberg_circuit`` for faster smoke tests without QCNN recursion depth.
"""

from __future__ import annotations

from typing import Iterator, List

import cirq
import numpy as np


def one_qubit_unitary(bit: cirq.Qid, variables: Iterator[float]) -> cirq.Circuit:
    return cirq.Circuit(
        cirq.X(bit) ** next(variables),
        cirq.Y(bit) ** next(variables),
        cirq.Z(bit) ** next(variables),
    )


def two_qubit_unitary(bits: list[cirq.Qid], variables: Iterator[float]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    circuit += one_qubit_unitary(bits[0], variables)
    circuit += one_qubit_unitary(bits[1], variables)
    circuit += [cirq.ZZ(*bits) ** next(variables)]
    circuit += [cirq.YY(*bits) ** next(variables)]
    circuit += [cirq.XX(*bits) ** next(variables)]
    circuit += one_qubit_unitary(bits[0], variables)
    circuit += one_qubit_unitary(bits[1], variables)
    return circuit


def two_qubit_pool(
    source_qubit: cirq.Qid, sink_qubit: cirq.Qid, variables: Iterator[float]
) -> cirq.Circuit:
    pool_circuit = cirq.Circuit()
    sink_basis_selector = one_qubit_unitary(sink_qubit, variables)
    source_basis_selector = one_qubit_unitary(source_qubit, variables)
    pool_circuit.append(sink_basis_selector)
    pool_circuit.append(source_basis_selector)
    pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
    pool_circuit.append(sink_basis_selector**-1)
    return pool_circuit


def quantum_conv_circuit(bits: list[cirq.Qid], variables: Iterator[float]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    for first, second in zip(bits[0::2], bits[1::2]):
        circuit += two_qubit_unitary([first, second], variables)
    for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
        circuit += two_qubit_unitary([first, second], variables)
    return circuit


def quantum_pool_circuit(
    source_bits: list[cirq.Qid], sink_bits: list[cirq.Qid], variables: Iterator[float]
) -> cirq.Circuit:
    circuit = cirq.Circuit()
    for source, sink in zip(source_bits, sink_bits):
        circuit += two_qubit_pool(source, sink, variables)
    return circuit


def quantum_full_circuit(qubits: list[cirq.Qid], variables: Iterator[float]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    circuit += [cirq.X(q) ** next(variables) for q in qubits]
    circuit += [cirq.Y(q) ** next(variables) for q in qubits]
    circuit += [cirq.X(q) ** next(variables) for q in qubits]
    if len(qubits) >= 2:
        circuit += [
            cirq.XX(q1, q2) ** next(variables)
            for q1, q2 in zip(qubits, qubits[1:] + [qubits[0]])
        ]
    circuit += [
        cirq.X(qubits[-1]) ** next(variables),
        cirq.Y(qubits[-1]) ** next(variables),
        cirq.X(qubits[-1]) ** next(variables),
    ]
    return circuit


def create_model_circuit(
    qubits: list[cirq.Qid],
    variables: Iterator[float],
    p: float = 0.0,
    noise_op=cirq.depolarize,
    mixed: bool = False,
    full_size: int = 4,
) -> cirq.Circuit:
    """Same recursion pattern as ``evaluate_qcnn_model.create_model_circuit``."""
    qnum = len(qubits)
    if qnum <= full_size:
        return quantum_full_circuit(qubits, variables)

    circuit = cirq.Circuit()
    circuit += quantum_conv_circuit(qubits, variables)
    if p > 1e-5:
        if mixed:
            circuit += cirq.bit_flip(p).on_each(*qubits[::3])
            circuit += cirq.depolarize(p).on_each(*qubits[1::3])
            circuit += cirq.phase_flip(p).on_each(*qubits[2::3])
        else:
            circuit += noise_op(p).on_each(*qubits)

    circuit += create_model_circuit(
        qubits[qnum // 2 :], variables, p=p, noise_op=noise_op, mixed=mixed, full_size=full_size
    )
    return circuit


def random_heisenberg_circuit(
    qubits: list[cirq.Qid],
    rng: np.random.Generator,
    depth: int,
    *,
    p_noise: float = 0.0,
    noise_op=cirq.depolarize,
) -> cirq.Circuit:
    """Light random brick: single-qubit rotations + ring CNOT + optional depolarizing."""
    circuit = cirq.Circuit()
    for _ in range(depth):
        for q in qubits:
            circuit.append(cirq.rx(rng.uniform(0, 2 * np.pi))(q))
            circuit.append(cirq.rz(rng.uniform(0, 2 * np.pi))(q))
        if len(qubits) >= 2:
            for i in range(len(qubits)):
                if rng.random() > 0.5:
                    a, b = qubits[i], qubits[(i + 1) % len(qubits)]
                    circuit.append(cirq.CNOT(a, b))
        if p_noise > 1e-12:
            for q in qubits:
                circuit.append(noise_op(p_noise)(q))
    return circuit


def random_variables() -> Iterator[float]:
    while True:
        yield float(np.random.rand())
