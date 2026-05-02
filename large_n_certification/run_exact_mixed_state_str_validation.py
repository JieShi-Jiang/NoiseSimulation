#!/usr/bin/env python3
"""
Exact small-n mixed-state STR validation for noisy QCNN / random circuits.

This entry point is intentionally small-n and dense:

- it simulates the full output density matrix with ``DensityMatrixSimulator``,
- it computes theorem-facing ``S/T/R`` ingredients from the eigendecomposition of ``rho``,
- and it records exact mixed-state primitives we will need before building a
  theorem-faithful ``tau4`` / noisy-ansatz ``tau5`` path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import cirq
import numpy as np

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from large_n_certification.circuits.qcnn_random import (
    create_model_circuit,
    random_heisenberg_circuit,
)
from large_n_certification.mixed_state_tools import summarize_str_partition

DEFAULT_LOG_ROOT = Path(_REPO_ROOT) / "logs"
NOISE_OPS = {
    "depolarizing": cirq.depolarize,
    "amplitude_damping": cirq.amplitude_damp,
    "bit_flip": cirq.bit_flip,
    "phase_flip": cirq.phase_flip,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nqubits", type=int, default=8)
    ap.add_argument("--nclasses", type=int, choices=(2, 4), default=4)
    ap.add_argument("--circuit", choices=("qcnn", "random"), default="qcnn")
    ap.add_argument("--noise-type", choices=tuple(NOISE_OPS.keys()), default="depolarizing")
    ap.add_argument("--noise-strength", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--random-depth", type=int, default=4)
    ap.add_argument("--full-size", type=int, default=None)
    ap.add_argument("--mixed-noise", action="store_true")
    ap.add_argument("--log-root", type=str, default=str(DEFAULT_LOG_ROOT))
    ap.add_argument("--run-tag", type=str, default="exact_mixed_state_str")
    return ap.parse_args()


def variables_from_seed(seed: int) -> Iterator[float]:
    rng = np.random.default_rng(int(seed))
    while True:
        yield float(rng.random())


def build_circuit(
    *,
    nqubits: int,
    circuit_name: str,
    seed: int,
    noise_type: str,
    noise_strength: float,
    random_depth: int,
    full_size: int | None,
    mixed_noise: bool,
) -> tuple[cirq.Circuit, list[cirq.GridQubit]]:
    qubits = sorted(cirq.GridQubit.rect(1, int(nqubits)))
    if str(circuit_name) == "random":
        rng = np.random.default_rng(int(seed))
        circuit = random_heisenberg_circuit(
            qubits,
            rng,
            depth=int(random_depth),
            p_noise=float(noise_strength),
            noise_op=NOISE_OPS[str(noise_type)],
        )
        return circuit, qubits

    circuit = create_model_circuit(
        qubits,
        variables_from_seed(int(seed)),
        p=float(noise_strength),
        noise_op=NOISE_OPS[str(noise_type)],
        mixed=bool(mixed_noise),
        full_size=max(4, int(full_size) if full_size is not None else int(nqubits) // 2 + 1),
    )
    return circuit, qubits


def simulate_density_matrix(circuit: cirq.Circuit, qubits: list[cirq.GridQubit]) -> np.ndarray:
    sim = cirq.DensityMatrixSimulator()
    result = sim.simulate(circuit, qubit_order=qubits)
    rho = np.asarray(result.final_density_matrix, dtype=np.complex128)
    return 0.5 * (rho + rho.conj().T)


def build_output_path(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d")
    root = Path(args.log_root)
    root.mkdir(parents=True, exist_ok=True)
    stem = (
        f"exact_mixed_state_str__run_tag={args.run_tag}__date={stamp}"
        f"__n={int(args.nqubits)}__c={int(args.nclasses)}__circuit={str(args.circuit)}"
    )
    return root / f"{stem}.json"


def main() -> None:
    args = parse_args()
    if int(args.nqubits) > 12:
        raise ValueError("This exact mixed-state STR validation is intentionally capped to n<=12.")

    t0 = time.time()
    circuit, qubits = build_circuit(
        nqubits=int(args.nqubits),
        circuit_name=str(args.circuit),
        seed=int(args.seed),
        noise_type=str(args.noise_type),
        noise_strength=float(args.noise_strength),
        random_depth=int(args.random_depth),
        full_size=args.full_size,
        mixed_noise=bool(args.mixed_noise),
    )
    rho = simulate_density_matrix(circuit, qubits)
    summary = summarize_str_partition(rho, qubits, n_classes=int(args.nclasses))
    elapsed = float(time.time() - t0)

    payload: dict[str, object] = {
        "run_tag": str(args.run_tag),
        "nqubits": int(args.nqubits),
        "nclasses": int(args.nclasses),
        "circuit": str(args.circuit),
        "noise_type": str(args.noise_type),
        "noise_strength": float(args.noise_strength),
        "seed": int(args.seed),
        "random_depth": int(args.random_depth),
        "full_size": None if args.full_size is None else int(args.full_size),
        "mixed_noise": bool(args.mixed_noise),
        "rho_dim": int(rho.shape[0]),
        "elapsed_seconds": elapsed,
        "tau4_status": "mixed_state_str_ready_constructive_upper_bound_pending",
        "tau5_status": "density_matrix_metric_pending_attack_loop",
        "notes": [
            "This runner validates exact mixed-state S/T/R ingredients from a dense rho.",
            "It is the intended bridge between the old pure-state tau4 shortcut and a theorem-faithful noisy-ansatz route.",
            "It does not yet construct the final tau4 upper bound or run a mixed-state tau5 attack search.",
        ],
        "str_summary": summary,
    }

    output_path = build_output_path(args)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"[done] n={int(args.nqubits)} c={int(args.nclasses)} circuit={str(args.circuit)} "
        f"noise={str(args.noise_type)} p={float(args.noise_strength):.4f} "
        f"pred={int(summary['predicted_index'])} competitor={int(summary['competitor_index'])} "
        f"rank={int(summary['effective_rank'])} elapsed={elapsed:.3f}s"
    )
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
