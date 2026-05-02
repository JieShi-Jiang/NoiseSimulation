#!/usr/bin/env python3
"""
Scalable ``tau1/tau2`` plus trajectory-low-rank mixed-state ``tau4/tau5`` for noisy QCNN / random circuits.

This runner targets the 20q / 25q scalability story:

- ``tau1`` and ``tau2`` still come from the existing TN spectral route,
- ``tau4`` is a constructive mixed-state upper bound on a low-rank trajectory approximation of ``rho``,
- ``tau5`` is an empirical three-attack state-space surrogate on the same low-rank support,
- and all outputs are explicitly labeled as trajectory/support-space approximations rather than exact dense-``rho`` theorem claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _bootstrap_cpu_threads_from_argv(default: int = 32) -> int:
    argv = sys.argv[1:]
    n = int(default)
    i = 0
    while i < len(argv):
        if argv[i] == "--threads" and i + 1 < len(argv):
            n = int(argv[i + 1])
            break
        if argv[i].startswith("--threads="):
            n = int(argv[i].split("=", 1)[1])
            break
        i += 1
    n = max(1, int(n))
    s = str(n)
    os.environ["OMP_NUM_THREADS"] = s
    os.environ["MKL_NUM_THREADS"] = s
    os.environ["OPENBLAS_NUM_THREADS"] = s
    os.environ["NUMEXPR_NUM_THREADS"] = s
    return n


_BOOTSTRAP_THREADS = _bootstrap_cpu_threads_from_argv()

import numpy as np

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from large_n_certification.estimation import probabilities_last_readout_basis_state_tn
from large_n_certification.run_noisy_qcnn_tau123_scalability import (
    DEFAULT_LOG_ROOT,
    NOISE_OPS,
    build_circuit,
    finalize_tau12_report,
    measurement_extrema,
    parse_basis_bits,
)
from large_n_certification.trajectory_low_rank_tau45 import (
    build_support_model_from_trajectories,
    class_probabilities_support,
    compute_tau4_support_multiclass,
    sample_trajectory_states,
    summarize_state_space_attacks,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nqubits", type=int, required=True)
    ap.add_argument("--nclasses", type=int, choices=(2, 4), required=True)
    ap.add_argument("--circuit", choices=("qcnn", "random"), default="qcnn")
    ap.add_argument("--noise-types", nargs="*", default=["depolarizing", "amplitude_damping", "bit_flip"])
    ap.add_argument("--noise-strengths", nargs="*", type=float, default=[0.02, 0.05, 0.10, 0.20, 0.30])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--random-depth", type=int, default=4)
    ap.add_argument("--niters", type=int, default=80)
    ap.add_argument("--full-size", type=int, default=None)
    ap.add_argument("--basis-bits", type=str, default=None)
    ap.add_argument("--mixed-noise", action="store_true")
    ap.add_argument("--log-root", type=str, default=str(DEFAULT_LOG_ROOT))
    ap.add_argument("--run-tag", type=str, default="tau1245_trajectory")
    ap.add_argument("--threads", type=int, default=_BOOTSTRAP_THREADS)
    ap.add_argument("--trajectory-shots", type=int, default=8)
    ap.add_argument("--support-basis-tol", type=float, default=1e-8)
    ap.add_argument("--attack-betas", nargs="*", type=float, default=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40])
    ap.add_argument("--random-trials", type=int, default=24)
    ap.add_argument("--pgd-steps", type=int, default=10)
    ap.add_argument("--pgd-step-size", type=float, default=0.08)
    ap.add_argument("--statevector-dtype", choices=("complex64", "complex128"), default="complex64")
    return ap.parse_args()


def build_output_path(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d")
    root = Path(args.log_root)
    root.mkdir(parents=True, exist_ok=True)
    stem = (
        f"tau1245_trajectory__run_tag={args.run_tag}__date={stamp}"
        f"__n={int(args.nqubits)}__c={int(args.nclasses)}__circuit={str(args.circuit)}"
    )
    return root / f"{stem}.json"


def main() -> None:
    args = parse_args()
    basis_bits = parse_basis_bits(args.basis_bits, int(args.nqubits))
    output_path = build_output_path(args)
    start_all = time.time()

    payload: dict[str, object] = {
        "run_tag": str(args.run_tag),
        "nqubits": int(args.nqubits),
        "nclasses": int(args.nclasses),
        "circuit": str(args.circuit),
        "seed": int(args.seed),
        "basis_bits": "".join(str(bit) for bit in basis_bits),
        "noise_types": [str(item) for item in args.noise_types],
        "noise_strengths": [float(v) for v in args.noise_strengths],
        "niters": int(args.niters),
        "mixed_noise": bool(args.mixed_noise),
        "threads": int(args.threads),
        "trajectory_shots": int(args.trajectory_shots),
        "support_basis_tol": float(args.support_basis_tol),
        "attack_betas": [float(v) for v in args.attack_betas],
        "random_trials": int(args.random_trials),
        "pgd_steps": int(args.pgd_steps),
        "pgd_step_size": float(args.pgd_step_size),
        "statevector_dtype": str(args.statevector_dtype),
        "cases": [],
        "notes": [
            "tau1/tau2 use the same TN spectral route as run_noisy_qcnn_tau123_scalability.",
            "tau4 uses a low-rank trajectory approximation of rho and a convex-mixture constructive state in support space.",
            "tau5 is a state-space empirical surrogate on the same support, with random / fgsm / pgd attack records.",
            "These tau4/tau5 values are trajectory-support approximations, not exact dense-rho theorem outputs.",
        ],
    }

    for noise_type in [str(item).strip().lower() for item in args.noise_types]:
        if noise_type not in NOISE_OPS:
            raise ValueError(f"Unsupported noise type {noise_type!r}. Supported: {sorted(NOISE_OPS)}")
        for strength in [float(v) for v in args.noise_strengths]:
            t0 = time.time()
            circuit, qubits = build_circuit(
                nqubits=int(args.nqubits),
                circuit_name=str(args.circuit),
                seed=int(args.seed),
                noise_type=noise_type,
                noise_strength=float(strength),
                random_depth=int(args.random_depth),
                full_size=args.full_size,
                mixed_noise=bool(args.mixed_noise),
            )
            clean_circuit, _ = build_circuit(
                nqubits=int(args.nqubits),
                circuit_name=str(args.circuit),
                seed=int(args.seed),
                noise_type=noise_type,
                noise_strength=0.0,
                random_depth=int(args.random_depth),
                full_size=args.full_size,
                mixed_noise=False,
            )

            noisy_probs = probabilities_last_readout_basis_state_tn(
                circuit,
                qubits,
                n_classes=int(args.nclasses),
                basis_bits=basis_bits,
            )
            clean_probs = probabilities_last_readout_basis_state_tn(
                clean_circuit,
                qubits,
                n_classes=int(args.nclasses),
                basis_bits=basis_bits,
            )
            extrema = measurement_extrema(
                circuit,
                qubits,
                n_classes=int(args.nclasses),
                niters=int(args.niters),
            )
            tau12 = finalize_tau12_report(
                noisy_probs=np.asarray(noisy_probs, dtype=np.float64),
                clean_probs=np.asarray(clean_probs, dtype=np.float64),
                extrema=extrema,
                noise_type=noise_type,
                noise_strength=float(strength),
            )

            states = sample_trajectory_states(
                circuit,
                qubits,
                basis_bits=basis_bits,
                shots=int(args.trajectory_shots),
                seed=int(args.seed) + int(round(1000.0 * float(strength))) + 7919 * len(payload["cases"]),
                dtype=str(args.statevector_dtype),
            )
            support_model = build_support_model_from_trajectories(
                states,
                n_classes=int(args.nclasses),
                basis_tol=float(args.support_basis_tol),
            )
            support_probs = class_probabilities_support(
                np.asarray(support_model["rho_support"], dtype=np.complex128),
                [np.asarray(op, dtype=np.complex128) for op in support_model["measurement_support"]],
            )
            tau4 = compute_tau4_support_multiclass(
                rho_support=np.asarray(support_model["rho_support"], dtype=np.complex128),
                measurement_support=[np.asarray(op, dtype=np.complex128) for op in support_model["measurement_support"]],
            )
            tau5 = summarize_state_space_attacks(
                rho_support=np.asarray(support_model["rho_support"], dtype=np.complex128),
                measurement_support=[np.asarray(op, dtype=np.complex128) for op in support_model["measurement_support"]],
                attack_betas=[float(v) for v in args.attack_betas],
                random_trials=int(args.random_trials),
                pgd_steps=int(args.pgd_steps),
                pgd_step_size=float(args.pgd_step_size),
                seed=int(args.seed) + 17041 * len(payload["cases"]),
            )

            elapsed = float(time.time() - t0)
            row = {
                "noise_type": str(noise_type),
                "noise_strength": float(strength),
                "elapsed_seconds": elapsed,
                "tau1": float(tau12["tau1"]),
                "tau2": float(tau12["tau2"]),
                "tau4": float(tau4["tau4"]),
                "tau5": float(tau5["tau5"]),
                "tau5_attack_found": bool(tau5["tau5_attack_found"]),
                "ranked_probabilities": list(tau12["ranked_probabilities"]),
                "ranked_outcome_indices": list(tau12["ranked_outcome_indices"]),
                "clean_probabilities": list(tau12["clean_probabilities"]),
                "noisy_probabilities": list(tau12["noisy_probabilities"]),
                "support_probabilities": support_probs.tolist(),
                "support_vs_tn_l1": float(np.sum(np.abs(np.asarray(support_probs, dtype=np.float64) - np.asarray(noisy_probs, dtype=np.float64)))),
                "support_rank": int(support_model["support_rank"]),
                "support_gram_eigenvalues_desc": list(support_model["gram_eigenvalues_desc"]),
                "str_summary_support": support_model["str_summary_support"],
                "tau12_report": tau12,
                "tau4_report": tau4,
                "tau5_report": tau5,
            }
            payload["cases"].append(row)
            print(
                f"[case] noise={noise_type} p={float(strength):.4f} "
                f"tau1={float(row['tau1']):.6f} tau2={float(row['tau2']):.6f} "
                f"tau4={float(row['tau4']):.6f} tau5={float(row['tau5']):.6f} "
                f"rank={int(row['support_rank'])} support_l1={float(row['support_vs_tn_l1']):.3e} "
                f"elapsed={elapsed:.2f}s"
            )

    payload["elapsed_seconds_total"] = float(time.time() - start_all)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
