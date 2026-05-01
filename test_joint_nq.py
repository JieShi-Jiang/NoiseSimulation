"""
Smoke test: random noisy circuit + joint Z-basis projectors on last two qubits (four outcomes).

Uses lipschitz(..., joint_measurement=P_k). Recommended env:

    conda activate qml_gpu
    python test_joint_nq.py --nqubits 6

Use --nqubits 4 locally if contraction is slow. Defaults favor lighter runs (--nqubits 6 --niters 80).
"""
from __future__ import annotations

import argparse
import numpy as np
import cirq

from qlipschitz import lipschitz


def random_noisy_circuit(qubits, rng, depth: int):
    circuit = cirq.Circuit()
    for _ in range(depth):
        for q in qubits:
            circuit.append(cirq.rx(rng.uniform(0, 2 * np.pi))(q))
            circuit.append(cirq.rz(rng.uniform(0, 2 * np.pi))(q))
        for i in range(len(qubits) - 1):
            if rng.random() > 0.5:
                circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        for q in qubits:
            circuit.append(cirq.depolarize(0.02).on(q))
    return circuit


def projector_comp_basis(k: int) -> np.ndarray:
    """Rank-1 projector |k><k|, k in {0,1,2,3} for basis order |00>,|01>,|10>,|11>."""
    p = np.zeros((4, 4), dtype=np.float64)
    p[k, k] = 1.0
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nqubits", type=int, default=6, help="total qubits (>=2); default 6 for lighter CPU runs")
    ap.add_argument("--depth", type=int, default=4, help="random circuit depth")
    ap.add_argument("--niters", type=int, default=80, help="power-iteration cap per eigen call")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.nqubits < 2:
        raise SystemExit("--nqubits must be at least 2 (joint measurement uses last two)")

    rng = np.random.default_rng(args.seed)
    qubits = cirq.GridQubit.rect(1, args.nqubits)
    circuit = random_noisy_circuit(sorted(qubits), rng, depth=args.depth)
    dummy_m = np.zeros((2, 2), dtype=np.float64)

    print(
        f"{args.nqubits}-qubit RX/RZ/CNOT + depolarize(0.02); "
        "joint readout on last two GridQubits"
    )
    print(f"depth={args.depth}, niters={args.niters}")
    print("Algorithm-1 style scalar e1 - e2 for each computational projector P_k\n")

    results = []
    for k in range(4):
        Pk = projector_comp_basis(k)
        k_val = lipschitz(
            circuit,
            list(qubits),
            dummy_m,
            joint_measurement=Pk,
            niters=args.niters,
        )
        results.append((k, float(k_val)))
        print(
            f"  outcome k={k} (|k><k| on last two qubits): K-like (e1-e2) = {k_val}\n"
        )

    print("Summary (k, value):", results)


if __name__ == "__main__":
    main()
