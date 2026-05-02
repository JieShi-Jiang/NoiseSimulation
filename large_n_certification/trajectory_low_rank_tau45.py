#!/usr/bin/env python3
"""Trajectory-based low-rank mixed-state tau4/tau5 surrogates for noisy circuits."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cirq
import numpy as np

from large_n_certification.mixed_state_tools import (
    density_matrix_trace_distance,
    projector_comp_basis,
    projector_labels,
    qubit_projector,
    summarize_str_partition_from_measurements,
)

EPS = 1e-12


def basis_bits_to_int(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return int(value)


def local_projectors(n_classes: int) -> list[np.ndarray]:
    if int(n_classes) == 2:
        return [qubit_projector(0), qubit_projector(1)]
    if int(n_classes) == 4:
        return [projector_comp_basis(idx, 2) for idx in range(4)]
    raise ValueError(f"Unsupported n_classes={n_classes}; expected 2 or 4.")


def simulate_trajectory_state(
    circuit: cirq.Circuit,
    qubits: list[cirq.Qid],
    *,
    basis_bits: list[int],
    seed: int,
    dtype: str = "complex64",
) -> np.ndarray:
    sim = cirq.Simulator(seed=int(seed))
    result = sim.simulate(
        circuit,
        qubit_order=qubits,
        initial_state=basis_bits_to_int(list(basis_bits)),
    )
    state = np.asarray(result.final_state_vector)
    return np.asarray(state, dtype=np.complex64 if dtype == "complex64" else np.complex128).reshape(-1)


def sample_trajectory_states(
    circuit: cirq.Circuit,
    qubits: list[cirq.Qid],
    *,
    basis_bits: list[int],
    shots: int,
    seed: int,
    dtype: str = "complex64",
) -> list[np.ndarray]:
    return [
        simulate_trajectory_state(
            circuit,
            qubits,
            basis_bits=list(basis_bits),
            seed=int(seed) + 1009 * shot,
            dtype=str(dtype),
        )
        for shot in range(int(shots))
    ]


@dataclass(frozen=True)
class SupportBasis:
    rank: int
    eigenvalues_gram: np.ndarray
    transform_v: np.ndarray
    inv_sqrt_gram: np.ndarray


def _support_basis_from_gram(gram: np.ndarray, *, tol: float = 1e-8) -> SupportBasis:
    gram_h = 0.5 * (np.asarray(gram, dtype=np.complex128) + np.asarray(gram, dtype=np.complex128).conj().T)
    evals, evecs = np.linalg.eigh(gram_h)
    order = np.argsort(np.real(evals))[::-1]
    evals = np.real(evals[order])
    evecs = evecs[:, order]
    keep = evals > float(tol)
    if not np.any(keep):
        raise ValueError("Trajectory Gram matrix is numerically rank-zero.")
    kept = evals[keep]
    vecs = evecs[:, keep]
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(kept, float(tol))))
    return SupportBasis(
        rank=int(kept.size),
        eigenvalues_gram=np.asarray(kept, dtype=np.float64),
        transform_v=np.asarray(vecs, dtype=np.complex128),
        inv_sqrt_gram=np.asarray(inv_sqrt, dtype=np.complex128),
    )


def _project_operator_to_support(op_gram: np.ndarray, basis: SupportBasis) -> np.ndarray:
    middle = basis.transform_v.conj().T @ np.asarray(op_gram, dtype=np.complex128) @ basis.transform_v
    projected = basis.inv_sqrt_gram @ middle @ basis.inv_sqrt_gram
    projected = 0.5 * (projected + projected.conj().T)
    return np.asarray(projected, dtype=np.complex128)


def _local_matrix_element(
    psi_left: np.ndarray,
    psi_right: np.ndarray,
    op_local: np.ndarray,
) -> complex:
    local_dim = int(op_local.shape[0])
    env_dim = int(psi_left.size // local_dim)
    left = np.asarray(psi_left, dtype=np.complex128).reshape(env_dim, local_dim)
    right = np.asarray(psi_right, dtype=np.complex128).reshape(env_dim, local_dim)
    return complex(np.einsum("ea,ab,eb->", left.conj(), op_local, right, optimize=True))


def build_support_model_from_trajectories(
    states: list[np.ndarray],
    *,
    n_classes: int,
    basis_tol: float = 1e-8,
) -> dict[str, object]:
    if not states:
        raise ValueError("Need at least one trajectory state.")

    labels = projector_labels(int(n_classes))
    projectors = local_projectors(int(n_classes))
    shot_count = int(len(states))
    gram = np.zeros((shot_count, shot_count), dtype=np.complex128)
    measurement_grams = [np.zeros((shot_count, shot_count), dtype=np.complex128) for _ in range(int(n_classes))]

    for i in range(shot_count):
        gram[i, i] = complex(np.vdot(states[i], states[i]))
        for k, proj in enumerate(projectors):
            measurement_grams[k][i, i] = _local_matrix_element(states[i], states[i], proj)
        for j in range(i):
            gij = complex(np.vdot(states[i], states[j]))
            gram[i, j] = gij
            gram[j, i] = np.conj(gij)
            for k, proj in enumerate(projectors):
                aij = _local_matrix_element(states[i], states[j], proj)
                measurement_grams[k][i, j] = aij
                measurement_grams[k][j, i] = np.conj(aij)

    basis = _support_basis_from_gram(gram, tol=float(basis_tol))
    rho_support = np.diag(np.asarray(basis.eigenvalues_gram, dtype=np.float64) / float(shot_count)).astype(np.complex128)
    measurement_support = [
        _project_operator_to_support(op_gram, basis)
        for op_gram in measurement_grams
    ]

    str_summary = summarize_str_partition_from_measurements(
        rho_support,
        measurement_support,
        labels=labels,
    )

    return {
        "labels": labels,
        "shots": shot_count,
        "support_rank": int(basis.rank),
        "gram_eigenvalues_desc": basis.eigenvalues_gram.tolist(),
        "rho_support": rho_support,
        "measurement_support": measurement_support,
        "str_summary_support": str_summary,
    }


def class_probabilities_support(rho_support: np.ndarray, measurement_support: list[np.ndarray]) -> np.ndarray:
    vals = [
        float(np.real(np.trace(np.asarray(op, dtype=np.complex128) @ np.asarray(rho_support, dtype=np.complex128))))
        for op in measurement_support
    ]
    arr = np.clip(np.asarray(vals, dtype=np.float64), 0.0, None)
    total = float(arr.sum())
    if total <= EPS:
        out = np.zeros_like(arr)
        out[0] = 1.0
        return out
    return arr / total


def _state_prediction(probs: np.ndarray) -> tuple[int, int]:
    ranking = sorted(enumerate(np.asarray(probs, dtype=np.float64).tolist()), key=lambda item: item[1], reverse=True)
    pred = int(ranking[0][0])
    comp = int(ranking[1][0]) if len(ranking) > 1 else int(pred)
    return pred, comp


def _pure_density(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm <= EPS:
        raise ValueError("Cannot build a pure density matrix from a zero vector.")
    v = v / norm
    return np.outer(v, v.conj())


def _candidate_pure_states(diff: np.ndarray, rho_support: np.ndarray) -> list[dict[str, object]]:
    diff = 0.5 * (np.asarray(diff, dtype=np.complex128) + np.asarray(diff, dtype=np.complex128).conj().T)
    rho_support = 0.5 * (np.asarray(rho_support, dtype=np.complex128) + np.asarray(rho_support, dtype=np.complex128).conj().T)

    devals, devecs = np.linalg.eigh(diff)
    positive_mask = np.real(devals) > 1e-10
    if not np.any(positive_mask):
        return []

    candidates: list[dict[str, object]] = []

    pos_space = devecs[:, positive_mask]
    evals_rho, evecs_rho = np.linalg.eigh(rho_support)
    principal = evecs_rho[:, int(np.argmax(np.real(evals_rho)))]
    projected = pos_space @ (pos_space.conj().T @ principal)
    if float(np.linalg.norm(projected)) > 1e-10:
        candidates.append({"method": "principal_rho_projected_to_positive_diff", "vector": projected})

    top_idx = int(np.argmax(np.real(devals)))
    candidates.append({"method": "top_positive_diff_eigenvector", "vector": devecs[:, top_idx]})

    for local_rank, col in enumerate(np.where(positive_mask)[0].tolist()):
        candidates.append({"method": f"positive_diff_eigenvector_{local_rank}", "vector": devecs[:, int(col)]})

    dedup: list[dict[str, object]] = []
    for item in candidates:
        vec = np.asarray(item["vector"], dtype=np.complex128).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm <= EPS:
            continue
        vec = vec / norm
        signature = np.abs(vec)
        if any(np.allclose(signature, np.abs(np.asarray(prev["vector"], dtype=np.complex128).reshape(-1)), atol=1e-8) for prev in dedup):
            continue
        dedup.append({"method": str(item["method"]), "vector": vec})
    return dedup


def _tau4_against_competitor_support(
    *,
    rho_support: np.ndarray,
    measurement_support: list[np.ndarray],
    predicted_index: int,
    competitor_index: int,
) -> dict[str, object]:
    rho_support = 0.5 * (np.asarray(rho_support, dtype=np.complex128) + np.asarray(rho_support, dtype=np.complex128).conj().T)
    op_c = np.asarray(measurement_support[int(predicted_index)], dtype=np.complex128)
    op_s = np.asarray(measurement_support[int(competitor_index)], dtype=np.complex128)
    y_c = float(np.real(np.trace(op_c @ rho_support)))
    y_s = float(np.real(np.trace(op_s @ rho_support)))
    gap = float(max(0.0, y_c - y_s))
    diff = 0.5 * ((op_s - op_c) + (op_s - op_c).conj().T)

    candidates = _candidate_pure_states(diff, rho_support)
    if not candidates:
        return {
            "tau4": 1.0,
            "tau4_method": "trivial_no_positive_competitor_space",
            "tau4_gap": gap,
            "tau4_beta": 0.0,
            "tau4_competitor": int(competitor_index),
            "tau4_state_trace_distance": 1.0,
            "tau4_misclassification_margin": 0.0,
            "tau4_candidates": [],
        }

    best: dict[str, object] | None = None
    reports: list[dict[str, object]] = []
    for item in candidates:
        phi = _pure_density(np.asarray(item["vector"], dtype=np.complex128))
        denom = float(np.real(np.trace(diff @ phi)))
        if denom <= 1e-12:
            continue
        beta = float(gap / denom) if gap > 0.0 else 0.0
        sigma = (rho_support + beta * phi) / (1.0 + beta)
        state_distance = density_matrix_trace_distance(rho_support, phi)
        tau4 = density_matrix_trace_distance(rho_support, sigma)
        sigma_yc = float(np.real(np.trace(sigma @ op_c)))
        sigma_ys = float(np.real(np.trace(sigma @ op_s)))
        report = {
            "tau4": float(tau4),
            "tau4_method": str(item["method"]),
            "tau4_gap": gap,
            "tau4_beta": beta,
            "tau4_competitor": int(competitor_index),
            "tau4_state_trace_distance": float(state_distance),
            "tau4_misclassification_margin": float(sigma_ys - sigma_yc),
            "tau4_denom": float(denom),
        }
        reports.append(report)
        if best is None or float(report["tau4"]) < float(best["tau4"]):
            best = report

    if best is None:
        return {
            "tau4": 1.0,
            "tau4_method": "degenerate_positive_space",
            "tau4_gap": gap,
            "tau4_beta": 0.0,
            "tau4_competitor": int(competitor_index),
            "tau4_state_trace_distance": 1.0,
            "tau4_misclassification_margin": 0.0,
            "tau4_candidates": reports,
        }

    out = dict(best)
    out["tau4_candidates"] = reports
    return out


def compute_tau4_support_multiclass(
    *,
    rho_support: np.ndarray,
    measurement_support: list[np.ndarray],
) -> dict[str, object]:
    probs = class_probabilities_support(rho_support, measurement_support)
    predicted_index, _ = _state_prediction(probs)
    best: dict[str, object] | None = None
    per_competitor: list[dict[str, object]] = []
    for competitor_index in range(len(measurement_support)):
        if int(competitor_index) == int(predicted_index):
            continue
        report = _tau4_against_competitor_support(
            rho_support=rho_support,
            measurement_support=measurement_support,
            predicted_index=int(predicted_index),
            competitor_index=int(competitor_index),
        )
        per_competitor.append(report)
        if best is None or float(report["tau4"]) < float(best["tau4"]):
            best = report

    if best is None:
        return {
            "tau4": 1.0,
            "tau4_method": "no_competitor",
            "tau4_gap": 0.0,
            "tau4_beta": 0.0,
            "tau4_competitor": None,
            "tau4_state_trace_distance": 1.0,
            "tau4_misclassification_margin": 0.0,
            "tau4_candidates": [],
            "per_competitor": [],
            "predicted_index": int(predicted_index),
            "predicted_probabilities": probs.tolist(),
        }

    out = dict(best)
    out["per_competitor"] = per_competitor
    out["predicted_index"] = int(predicted_index)
    out["predicted_probabilities"] = probs.tolist()
    out["tau4_mode"] = "trajectory_low_rank_support_convex_mixture"
    return out


def _attack_success(
    sigma: np.ndarray,
    *,
    measurement_support: list[np.ndarray],
    orig_pred: int,
) -> tuple[bool, int, np.ndarray]:
    probs = class_probabilities_support(sigma, measurement_support)
    pred, _ = _state_prediction(probs)
    return bool(int(pred) != int(orig_pred)), int(pred), probs


def _mix_with_pure_state(rho_support: np.ndarray, pure_state: np.ndarray, beta: float) -> np.ndarray:
    beta = float(max(0.0, beta))
    return (np.asarray(rho_support, dtype=np.complex128) + beta * np.asarray(pure_state, dtype=np.complex128)) / (1.0 + beta)


def _random_pure_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    real = rng.normal(size=int(dim))
    imag = rng.normal(size=int(dim))
    vec = np.asarray(real + 1j * imag, dtype=np.complex128)
    vec = vec / max(float(np.linalg.norm(vec)), EPS)
    return vec


def summarize_state_space_attacks(
    *,
    rho_support: np.ndarray,
    measurement_support: list[np.ndarray],
    attack_betas: list[float],
    random_trials: int,
    pgd_steps: int,
    pgd_step_size: float,
    seed: int,
) -> dict[str, object]:
    rho_support = 0.5 * (np.asarray(rho_support, dtype=np.complex128) + np.asarray(rho_support, dtype=np.complex128).conj().T)
    orig_probs = class_probabilities_support(rho_support, measurement_support)
    orig_pred, orig_runner = _state_prediction(orig_probs)
    dim = int(rho_support.shape[0])
    rng = np.random.default_rng(int(seed))

    random_records: list[dict[str, object]] = []
    random_best: float | None = None
    for beta in [float(v) for v in attack_betas]:
        for trial in range(int(random_trials)):
            pure = _pure_density(_random_pure_state(dim, rng))
            sigma = _mix_with_pure_state(rho_support, pure, beta)
            success, adv_pred, probs = _attack_success(sigma, measurement_support=measurement_support, orig_pred=orig_pred)
            dist = density_matrix_trace_distance(rho_support, sigma)
            random_records.append(
                {
                    "beta": float(beta),
                    "trial": int(trial),
                    "success": bool(success),
                    "adv_pred": int(adv_pred),
                    "trace_distance": float(dist),
                    "probabilities": probs.tolist(),
                }
            )
            if success and (random_best is None or float(dist) < float(random_best)):
                random_best = float(dist)

    fgsm_records: list[dict[str, object]] = []
    fgsm_best: float | None = None
    for competitor in range(len(measurement_support)):
        if int(competitor) == int(orig_pred):
            continue
        diff = 0.5 * (
            (np.asarray(measurement_support[int(competitor)], dtype=np.complex128) - np.asarray(measurement_support[int(orig_pred)], dtype=np.complex128))
            + (np.asarray(measurement_support[int(competitor)], dtype=np.complex128) - np.asarray(measurement_support[int(orig_pred)], dtype=np.complex128)).conj().T
        )
        evals, evecs = np.linalg.eigh(diff)
        top_idx = int(np.argmax(np.real(evals)))
        if float(np.real(evals[top_idx])) <= 1e-10:
            continue
        pure = _pure_density(evecs[:, top_idx])
        for beta in [float(v) for v in attack_betas]:
            sigma = _mix_with_pure_state(rho_support, pure, beta)
            success, adv_pred, probs = _attack_success(sigma, measurement_support=measurement_support, orig_pred=orig_pred)
            dist = density_matrix_trace_distance(rho_support, sigma)
            fgsm_records.append(
                {
                    "beta": float(beta),
                    "competitor": int(competitor),
                    "success": bool(success),
                    "adv_pred": int(adv_pred),
                    "trace_distance": float(dist),
                    "probabilities": probs.tolist(),
                    "gradient_mode": "support_top_diff_eigenvector",
                }
            )
            if success and (fgsm_best is None or float(dist) < float(fgsm_best)):
                fgsm_best = float(dist)

    pgd_records: list[dict[str, object]] = []
    pgd_best: float | None = None
    sigma = np.asarray(rho_support, dtype=np.complex128).copy()
    step_size = float(max(0.0, pgd_step_size))
    for step in range(int(pgd_steps)):
        probs = class_probabilities_support(sigma, measurement_support)
        competitor = int(np.argsort(probs)[::-1][1]) if probs.size > 1 else int(orig_runner)
        diff = 0.5 * (
            (np.asarray(measurement_support[int(competitor)], dtype=np.complex128) - np.asarray(measurement_support[int(orig_pred)], dtype=np.complex128))
            + (np.asarray(measurement_support[int(competitor)], dtype=np.complex128) - np.asarray(measurement_support[int(orig_pred)], dtype=np.complex128)).conj().T
        )
        evals, evecs = np.linalg.eigh(diff)
        top_idx = int(np.argmax(np.real(evals)))
        if float(np.real(evals[top_idx])) <= 1e-10:
            break
        pure = _pure_density(evecs[:, top_idx])
        sigma = _mix_with_pure_state(sigma, pure, step_size / max(EPS, 1.0 - step_size))
        sigma = 0.5 * (sigma + sigma.conj().T)
        sigma /= max(EPS, float(np.real(np.trace(sigma))))
        success, adv_pred, probs_new = _attack_success(sigma, measurement_support=measurement_support, orig_pred=orig_pred)
        dist = density_matrix_trace_distance(rho_support, sigma)
        pgd_records.append(
            {
                "step": int(step + 1),
                "competitor": int(competitor),
                "success": bool(success),
                "adv_pred": int(adv_pred),
                "trace_distance": float(dist),
                "probabilities": probs_new.tolist(),
                "step_size": float(step_size),
            }
        )
        if success and (pgd_best is None or float(dist) < float(pgd_best)):
            pgd_best = float(dist)

    finite = [value for value in (random_best, fgsm_best, pgd_best) if value is not None and np.isfinite(value)]
    tau5_raw = None if not finite else float(min(finite))
    return {
        "attack_mode": "trajectory_support_state_space",
        "orig_pred": int(orig_pred),
        "orig_runner_up": int(orig_runner),
        "orig_probabilities": orig_probs.tolist(),
        "tau_random": 1.0 if random_best is None else float(random_best),
        "tau_fgsm": 1.0 if fgsm_best is None else float(fgsm_best),
        "tau_pgd": 1.0 if pgd_best is None else float(pgd_best),
        "tau5_raw": tau5_raw,
        "tau5": 1.0 if tau5_raw is None else float(tau5_raw),
        "tau5_attack_found": bool(tau5_raw is not None),
        "random": random_records,
        "fgsm": fgsm_records,
        "pgd": pgd_records,
    }
