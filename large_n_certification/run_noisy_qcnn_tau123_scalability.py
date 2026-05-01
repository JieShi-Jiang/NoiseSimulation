#!/usr/bin/env python3
"""
Scalable ``tau1/tau2`` and depolarizing-only ``tau3`` for noisy QCNN / random circuits.

This runner is intentionally conservative:

- it supports **noisy ansatz** circuits at 20/25/27 qubits,
- it computes class probabilities ``y_k`` either by exact density matrix (small ``n``)
  or by tensor-network contraction of ``<b| E^dagger(P_k) |b>`` for a computational-basis
  product input,
- it computes spectral extrema of ``E^dagger(P_k)`` with the existing TN + power-iteration
  route from ``qlipschitz.py``,
- it reports ``tau1`` and ``tau2`` for all supported noise channels,
- it reports ``tau3`` only for depolarizing noise,
- and it records explicit status fields for ``tau4`` / ``tau5`` instead of pretending
  that mixed-state large-``n`` upper bounds are already solved.

Examples
--------

    cd /home/young_isacs/work/NoiseSimulation
    python -m large_n_certification.run_noisy_qcnn_tau123_scalability \
      --nqubits 20 --nclasses 4 --noise-types depolarizing amplitude_damping bit_flip

    python -m large_n_certification.run_noisy_qcnn_tau123_scalability \
      --nqubits 27 --nclasses 4 --noise-types bit_flip --noise-strengths 0.01 --niters 80
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import cirq
import numpy as np

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qlipschitz import largest_eigenvalue, model_to_mv, smallest_eigenvalue

from large_n_certification.circuits.qcnn_random import (
    create_model_circuit,
    random_heisenberg_circuit,
)
from large_n_certification.estimation import (
    probabilities_last_readout_basis_state_tn,
    probabilities_last_readout_exact_dm,
    probabilities_last_readout_noiseless,
    projector_comp_basis,
    qubit_projector,
)

EPS = 1e-12
DEFAULT_LOG_ROOT = Path(_REPO_ROOT) / "logs"
NOISE_OPS = {
    "depolarizing": cirq.depolarize,
    "amplitude_damping": cirq.amplitude_damp,
    "bit_flip": cirq.bit_flip,
    "phase_flip": cirq.phase_flip,
}


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
    ap.add_argument("--max-dm", type=int, default=12)
    ap.add_argument("--basis-bits", type=str, default=None, help="Computational-basis input, e.g. 0000 or 0101.")
    ap.add_argument("--mixed-noise", action="store_true", help="Use mixed bit-flip/depolarizing/phase-flip placement in QCNN mode.")
    ap.add_argument("--log-root", type=str, default=str(DEFAULT_LOG_ROOT))
    ap.add_argument("--run-tag", type=str, default="tau123_scalability")
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

    vars_iter = variables_from_seed(int(seed))
    circuit = create_model_circuit(
        qubits,
        vars_iter,
        p=float(noise_strength),
        noise_op=NOISE_OPS[str(noise_type)],
        mixed=bool(mixed_noise),
        full_size=max(4, int(full_size) if full_size is not None else int(nqubits) // 2 + 1),
    )
    return circuit, qubits


def parse_basis_bits(bits: str | None, nqubits: int) -> list[int]:
    if bits is None:
        return [0] * int(nqubits)
    clean = str(bits).strip()
    if len(clean) != int(nqubits):
        raise ValueError(f"--basis-bits length {len(clean)} does not match nqubits={nqubits}")
    parsed = [0 if ch == "0" else 1 if ch == "1" else -1 for ch in clean]
    if any(bit < 0 for bit in parsed):
        raise ValueError(f"--basis-bits must be a binary string, got {bits!r}")
    return parsed


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(values, dtype=np.float64).reshape(-1), 0.0, None)
    total = float(probs.sum())
    if total <= EPS:
        out = np.zeros_like(probs)
        out[0] = 1.0
        return out
    return probs / total


def safe_tau_value(num: float, den: float, eps: float = EPS) -> float:
    num = max(0.0, float(num))
    den = max(eps, float(den))
    return float(num / den)


def safe_condition_ratio(lam_max: float, lam_min: float, eps: float = EPS) -> float:
    if abs(float(lam_min)) <= eps:
        return float("inf")
    return float(lam_max / lam_min)


def safe_bound(num: float, kappa: float, eps: float = EPS) -> float:
    num = max(0.0, float(num))
    if not np.isfinite(float(kappa)):
        return 0.0
    return float(num / max(eps, float(kappa)))


def compute_tau3(clean_probs: np.ndarray, p: float) -> float:
    clean_probs = normalize_probabilities(clean_probs)
    ordered = sorted(clean_probs.tolist(), reverse=True)
    y0_clean = float(max(ordered[0], EPS))
    y1_clean = float(max(ordered[1], EPS))
    n_classes = int(clean_probs.shape[0])
    beta = float(p / max(EPS, n_classes * (1.0 - p)))
    prefactor = (math.sqrt(y0_clean + beta) - math.sqrt(y1_clean + beta)) / math.sqrt(y1_clean + beta)
    return float(max(0.0, prefactor * beta))


def measurement_extrema(
    circuit: cirq.Circuit,
    qubits: list[cirq.GridQubit],
    *,
    n_classes: int,
    niters: int,
) -> dict[str, dict[str, float]]:
    dummy_binary = np.zeros((2, 2), dtype=np.complex128)
    results: dict[str, dict[str, float]] = {}
    for outcome in range(int(n_classes)):
        if int(n_classes) == 2:
            measurement = qubit_projector(outcome)
            _, mv1, mv2 = model_to_mv(circuit, qubits, measurement, joint_measurement=None)
        else:
            measurement = dummy_binary
            joint_measurement = projector_comp_basis(outcome)
            _, mv1, mv2 = model_to_mv(circuit, qubits, measurement, joint_measurement=joint_measurement)
        lam_max_raw = float(largest_eigenvalue(len(qubits), mv1, int(niters)))
        lam_min_raw = float(smallest_eigenvalue(len(qubits), mv2, int(niters)))
        lam_max = min(1.0, max(0.0, lam_max_raw))
        lam_min = min(1.0, max(0.0, lam_min_raw))
        if abs(lam_min) <= 1e-8:
            lam_min = 0.0
        if abs(1.0 - lam_max) <= 1e-8:
            lam_max = 1.0
        if lam_min > lam_max:
            lam_min = lam_max
        delta = float(max(0.0, lam_max - lam_min))
        results[f"povm_{outcome}"] = {
            "measurement_index": int(outcome),
            "min_eigenvalue": lam_min,
            "max_eigenvalue": lam_max,
            "delta": delta,
            "min_eigenvalue_raw": lam_min_raw,
            "max_eigenvalue_raw": lam_max_raw,
        }
    return results


def finalize_tau12_report(
    *,
    noisy_probs: np.ndarray,
    clean_probs: np.ndarray,
    extrema: dict[str, dict[str, float]],
    noise_type: str,
    noise_strength: float,
) -> dict[str, object]:
    probs = normalize_probabilities(noisy_probs)
    clean_probs = normalize_probabilities(clean_probs)
    ranking = sorted(enumerate(probs.tolist()), key=lambda item: item[1], reverse=True)
    ranked_indices = [int(idx) for idx, _ in ranking]
    ranked_probs = [float(prob) for _, prob in ranking]
    padded = ranked_probs + [0.0] * max(0, 4 - len(ranked_probs))
    y0 = float(max(padded[0], EPS))
    y1 = float(max(padded[1], EPS))
    ratio = float(y0 / y1)
    sqrt_ratio = float(math.sqrt(ratio))
    inv_sqrt_ratio = 1.0 / max(sqrt_ratio, EPS)

    r1_candidates: list[float] = []
    condition_numbers: list[float] = []
    per_measurement: list[dict[str, object]] = []

    for rank, outcome_idx in enumerate(ranked_indices):
        eig = extrema[f"povm_{outcome_idx}"]
        lam_min = float(eig["min_eigenvalue"])
        lam_max = float(eig["max_eigenvalue"])
        delta = float(eig["delta"])
        y_k = float(ranked_probs[rank])

        tau0_threshold = float(y_k * inv_sqrt_ratio)
        tau1_threshold = float(y_k * sqrt_ratio)
        tau0_condition = bool(lam_min <= tau0_threshold + EPS)
        tau1_condition = bool(lam_max + EPS >= tau1_threshold)

        tau_k_0 = safe_tau_value(y_k * (1.0 - inv_sqrt_ratio), delta) if tau0_condition else 1.0
        tau_k_1 = safe_tau_value(y_k * (sqrt_ratio - 1.0), delta) if tau1_condition else 1.0
        kappa = safe_condition_ratio(lam_max, lam_min)

        r1_candidates.extend([tau_k_0, tau_k_1])
        condition_numbers.append(kappa)
        per_measurement.append(
            {
                "rank": int(rank),
                "measurement_index": int(outcome_idx),
                "y_k": y_k,
                "lambda_k_m": lam_min,
                "lambda_k_M": lam_max,
                "Delta_k": delta,
                "kappa_k": None if not np.isfinite(kappa) else float(kappa),
                "kappa_k_is_infinite": bool(not np.isfinite(kappa)),
                "tau_k_0_condition": bool(tau0_condition),
                "tau_k_1_condition": bool(tau1_condition),
                "tau_k_0": float(tau_k_0),
                "tau_k_1": float(tau_k_1),
                "pairwise_min_tau": float(min(tau_k_0, tau_k_1)),
            }
        )

    kappa_star = float(max(condition_numbers)) if condition_numbers else float("inf")
    tau1 = float(min(r1_candidates)) if r1_candidates else 0.0
    tau2 = float(safe_bound(sqrt_ratio - 1.0, kappa_star))
    tau3 = None
    if str(noise_type) == "depolarizing":
        tau3 = float(compute_tau3(clean_probs, float(noise_strength)))

    return {
        "ranked_outcome_indices": ranked_indices,
        "ranked_probabilities": ranked_probs,
        "clean_probabilities": clean_probs.tolist(),
        "noisy_probabilities": probs.tolist(),
        "y0": float(padded[0]),
        "y1": float(padded[1]),
        "y2": float(padded[2]),
        "y3": float(padded[3]),
        "R_rho": ratio,
        "sqrt_y0_over_y1": sqrt_ratio,
        "tau1": tau1,
        "tau2": tau2,
        "tau3": tau3,
        "kappa_star": None if not np.isfinite(kappa_star) else float(kappa_star),
        "kappa_star_is_infinite": bool(not np.isfinite(kappa_star)),
        "per_measurement": per_measurement,
        "tau4_status": "not_implemented_for_large_n_mixed_state",
        "tau5_status": "not_implemented_for_noisy_ansatz_mixed_state_attack",
    }


def build_output_path(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d")
    root = Path(args.log_root)
    root.mkdir(parents=True, exist_ok=True)
    stem = (
        f"tau123_scalability__run_tag={args.run_tag}__date={stamp}"
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
        "max_dm": int(args.max_dm),
        "mixed_noise": bool(args.mixed_noise),
        "cases": [],
        "notes": [
            "tau1/tau2 use noisy-ansatz effective measurement extrema via TN + power iteration.",
            "tau3 is reported only for depolarizing noise.",
            "tau4/tau5 are intentionally marked unresolved here for mixed-state noisy-ansatz large-n runs.",
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

            if int(args.nqubits) <= int(args.max_dm):
                noisy_probs = probabilities_last_readout_exact_dm(circuit, qubits, n_classes=int(args.nclasses))
                prob_method = "exact_density_matrix"
            else:
                noisy_probs = probabilities_last_readout_basis_state_tn(
                    circuit,
                    qubits,
                    n_classes=int(args.nclasses),
                    basis_bits=basis_bits,
                )
                prob_method = "tn_effective_measurement_basis_state"

            clean_probs = probabilities_last_readout_noiseless(clean_circuit, qubits, n_classes=int(args.nclasses))
            extrema = measurement_extrema(circuit, qubits, n_classes=int(args.nclasses), niters=int(args.niters))
            report = finalize_tau12_report(
                noisy_probs=noisy_probs,
                clean_probs=clean_probs,
                extrema=extrema,
                noise_type=noise_type,
                noise_strength=float(strength),
            )
            report["noise_type"] = str(noise_type)
            report["noise_strength"] = float(strength)
            report["probability_method"] = prob_method
            report["elapsed_seconds"] = float(time.time() - t0)
            payload["cases"].append(report)

            tau3_str = "n/a" if report["tau3"] is None else f"{float(report['tau3']):.6f}"
            print(
                f"[tau123_scalability] n={args.nqubits} c={args.nclasses} noise={noise_type} p={strength:.3f} "
                f"| method={prob_method} | tau1={float(report['tau1']):.6f} "
                f"| tau2={float(report['tau2']):.6f} | tau3={tau3_str} "
                f"| elapsed={float(report['elapsed_seconds']):.2f}s"
            )

    payload["output_json"] = str(output_path)
    payload["total_elapsed_seconds"] = float(time.time() - start_all)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[tau123_scalability] wrote {output_path}")


if __name__ == "__main__":
    main()
