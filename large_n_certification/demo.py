#!/usr/bin/env python3
"""Demonstrate large-n certification hooks (QCNN / random circuits)."""

from __future__ import annotations

import argparse
import os
import sys
import time

import cirq
import numpy as np

# Allow `python large_n_certification/demo.py` from NoiseSimulation/
_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nqubits", type=int, default=8)
    ap.add_argument("--circuit", choices=("qcnn", "random"), default="qcnn")
    ap.add_argument("--noise-p", type=float, default=0.01)
    ap.add_argument("--mixed-noise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-dm", type=int, default=14, help="max qubits for density-matrix noisy branch")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    qubits = sorted(cirq.GridQubit.rect(1, args.nqubits))

    from large_n_certification.circuits.qcnn_random import (
        create_model_circuit,
        random_heisenberg_circuit,
        random_variables,
    )
    from large_n_certification.estimation import (
        yk_exact_dm_binary_last,
        yk_noiseless_binary_last_qubit,
    )
    from large_n_certification.scheme import LargeNCertificationPlan

    plan = LargeNCertificationPlan()
    vars_iter = random_variables()

    t0 = time.time()
    if args.circuit == "qcnn":
        circuit = create_model_circuit(
            qubits,
            vars_iter,
            p=args.noise_p,
            noise_op=cirq.depolarize,
            mixed=args.mixed_noise,
            full_size=max(4, args.nqubits // 2 + 1),
        )
    else:
        circuit = random_heisenberg_circuit(
            qubits, rng, depth=4, p_noise=args.noise_p, noise_op=cirq.depolarize
        )

    y_clean = yk_noiseless_binary_last_qubit(circuit, qubits)
    print(f"circuit={args.circuit}, n={args.nqubits}, noise_p={args.noise_p}")
    print(f"plan max_exact_dm_qubits={plan.max_exact_density_matrix_qubits}")
    print(f"P(last=|0>) noiseless surrogate (strip channels): {y_clean:.6f}")

    if args.nqubits <= args.max_dm:
        try:
            y_noisy = yk_exact_dm_binary_last(circuit, qubits)
            print(f"P(last=|0>) density-matrix (noisy):              {y_noisy:.6f}")
        except MemoryError:
            print("density-matrix branch OOM — reduce --nqubits or --max-dm")
    else:
        print(
            f"skip density-matrix noisy simulation (n>{args.max_dm}); "
            "use trajectory MC or hardware (future)."
        )

    # Optional: spectral reference from qlipschitz (binary projector), same as evaluate_qcnn_model
    try:
        from qlipschitz import lipschitz

        m = np.array([[1.0, 0.0], [0.0, 0.0]])
        k_lip = lipschitz(circuit, list(qubits), m, niters=60)
        print(f"lipschitz(circuit, |0><0| last qubit) niters=60: K-like = {k_lip}")
    except Exception as exc:  # noqa: BLE001
        print(f"lipschitz skipped: {exc}")

    print(f"elapsed {time.time() - t0:.3f}s")


if __name__ == "__main__":
    main()
