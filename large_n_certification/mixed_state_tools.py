#!/usr/bin/env python3
"""Exact mixed-state helpers for small-n theorem-faithful validation."""

from __future__ import annotations

import cirq
import numpy as np

EPS = 1e-12


def projector_labels(n_classes: int) -> list[str]:
    if int(n_classes) == 2:
        return ["|0>", "|1>"]
    if int(n_classes) == 4:
        return ["|00>", "|01>", "|10>", "|11>"]
    raise ValueError(f"Unsupported n_classes={n_classes}; expected 2 or 4.")


def qubit_projector(bit: int) -> np.ndarray:
    if int(bit) == 0:
        return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    if int(bit) == 1:
        return np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    raise ValueError(f"Unsupported qubit projector bit={bit}; expected 0 or 1.")


def projector_comp_basis(index: int, n_readout: int) -> np.ndarray:
    if int(index) < 0 or int(index) >= 2 ** int(n_readout):
        raise ValueError(f"Outcome index {index} out of range for n_readout={n_readout}.")
    bits = format(int(index), f"0{int(n_readout)}b")
    proj = np.array([[1.0]], dtype=np.complex128)
    for bit_char in bits:
        proj = np.kron(proj, qubit_projector(int(bit_char)))
    return proj


def readout_qubit_count(n_classes: int) -> int:
    if int(n_classes) == 2:
        return 1
    if int(n_classes) == 4:
        return 2
    raise ValueError(f"Unsupported n_classes={n_classes}; expected 2 or 4.")


def projector_last_readout_computational(
    outcome_index: int,
    sorted_qubits: list[cirq.GridQubit],
    *,
    n_classes: int,
) -> np.ndarray:
    n_readout = readout_qubit_count(int(n_classes))
    if len(sorted_qubits) < n_readout:
        raise ValueError(
            f"Need at least {n_readout} qubits for n_classes={n_classes}, got {len(sorted_qubits)}."
        )
    readout_proj = projector_comp_basis(int(outcome_index), n_readout)
    mats = [np.eye(2, dtype=np.complex128) for _ in sorted_qubits[:-n_readout]]
    for bit_char in format(int(outcome_index), f"0{n_readout}b"):
        mats.append(qubit_projector(int(bit_char)))
    op = mats[0]
    for mat in mats[1:]:
        op = np.kron(op, mat)
    return op


def probabilities_last_readout_from_rho(
    rho: np.ndarray,
    sorted_qubits: list[cirq.GridQubit],
    *,
    n_classes: int,
) -> np.ndarray:
    probs = []
    for outcome_index in range(int(n_classes)):
        proj = projector_last_readout_computational(
            outcome_index,
            sorted_qubits,
            n_classes=int(n_classes),
        )
        probs.append(float(np.real(np.trace(proj @ rho))))
    probs_arr = np.asarray(probs, dtype=np.float64)
    probs_arr = np.clip(probs_arr, 0.0, None)
    total = float(probs_arr.sum())
    if total <= EPS:
        out = np.zeros(int(n_classes), dtype=np.float64)
        out[0] = 1.0
        return out
    return probs_arr / total


def density_matrix_trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    delta = np.asarray(rho, dtype=np.complex128) - np.asarray(sigma, dtype=np.complex128)
    delta = 0.5 * (delta + delta.conj().T)
    eigvals = np.linalg.eigvalsh(delta)
    return float(0.5 * np.sum(np.abs(np.real(eigvals))))


def str_partition_from_measurements(
    rho: np.ndarray,
    measurement_ops: list[np.ndarray],
    c: int,
    s: int,
    *,
    ev_tol: float = 1e-8,
    cmp_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
    rho = np.asarray(rho, dtype=np.complex128)
    rho = 0.5 * (rho + rho.conj().T)
    dim = rho.shape[0]
    w, V = np.linalg.eigh(rho)
    idx_desc = np.argsort(np.real(w))[::-1]
    w = np.real(w[idx_desc])
    V = V[:, idx_desc]

    floor = max(float(ev_tol), float(np.max(w)) * 1e-12 if w.size else float(ev_tol))
    Pc = np.asarray(measurement_ops[int(c)], dtype=np.complex128)
    Ps = np.asarray(measurement_ops[int(s)], dtype=np.complex128)

    Z = np.zeros_like(rho)
    S, T, R = Z.copy(), Z.copy(), Z.copy()
    ledger: list[dict[str, float | int | str]] = []

    for ell in range(dim):
        lam = float(w[ell])
        if lam < floor:
            continue
        phi = V[:, ell]
        pc = float(np.real(phi.conj().T @ Pc @ phi))
        ps = float(np.real(phi.conj().T @ Ps @ phi))
        piece = lam * np.outer(phi, phi.conj())

        if pc > ps + cmp_tol:
            bucket, target = "S", S
        elif pc < ps - cmp_tol:
            bucket, target = "T", T
        else:
            bucket, target = "R", R

        target += piece
        ledger.append(
            {
                "eigen_index_desc": int(ell),
                "eigenvalue": lam,
                "bucket": bucket,
                "pc": pc,
                "ps": ps,
            }
        )

    return S, T, R, ledger


def str_partition_mixed(
    rho: np.ndarray,
    sorted_qubits: list[cirq.GridQubit],
    c: int,
    s: int,
    *,
    n_classes: int,
    ev_tol: float = 1e-8,
    cmp_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
    measurement_ops = [
        projector_last_readout_computational(
            outcome_index,
            sorted_qubits,
            n_classes=int(n_classes),
        )
        for outcome_index in range(int(n_classes))
    ]
    return str_partition_from_measurements(
        rho,
        measurement_ops,
        c,
        s,
        ev_tol=float(ev_tol),
        cmp_tol=float(cmp_tol),
    )


def summarize_str_partition_from_measurements(
    rho: np.ndarray,
    measurement_ops: list[np.ndarray],
    *,
    labels: list[str] | None = None,
    ev_tol: float = 1e-8,
    cmp_tol: float = 1e-9,
) -> dict[str, object]:
    rho = np.asarray(rho, dtype=np.complex128)
    rho = 0.5 * (rho + rho.conj().T)
    probs = []
    for op in measurement_ops:
        probs.append(float(np.real(np.trace(np.asarray(op, dtype=np.complex128) @ rho))))
    probs_arr = np.asarray(probs, dtype=np.float64)
    probs_arr = np.clip(probs_arr, 0.0, None)
    total = float(probs_arr.sum())
    if total <= EPS:
        probs_arr = np.zeros_like(probs_arr)
        probs_arr[0] = 1.0
    else:
        probs_arr = probs_arr / total

    ranking = sorted(enumerate(probs_arr.tolist()), key=lambda item: item[1], reverse=True)
    ranked_indices = [int(idx) for idx, _ in ranking]
    ranked_probabilities = [float(prob) for _, prob in ranking]
    predicted_index = int(ranked_indices[0])
    competitor_index = int(ranked_indices[1]) if len(ranked_indices) > 1 else int(predicted_index)

    S, T, R, ledger = str_partition_from_measurements(
        rho,
        measurement_ops,
        predicted_index,
        competitor_index,
        ev_tol=float(ev_tol),
        cmp_tol=float(cmp_tol),
    )

    eigvals = np.real(np.linalg.eigvalsh(rho))
    eigvals_desc = np.sort(eigvals)[::-1]
    s_norm = float(np.linalg.norm(S, "fro"))
    t_norm = float(np.linalg.norm(T, "fro"))
    r_norm = float(np.linalg.norm(R, "fro"))
    reconstruction_error = float(np.linalg.norm(S + T + R - rho, "fro"))
    nonempty_bucket_count = int(sum(1 for value in (s_norm, t_norm, r_norm) if value > 1e-6))

    return {
        "n_classes": int(len(measurement_ops)),
        "labels": list(labels) if labels is not None else [f"class_{idx}" for idx in range(len(measurement_ops))],
        "probabilities": probs_arr.tolist(),
        "ranked_outcome_indices": ranked_indices,
        "ranked_probabilities": ranked_probabilities,
        "predicted_index": predicted_index,
        "competitor_index": competitor_index,
        "top_gap": float(ranked_probabilities[0] - ranked_probabilities[1]) if len(ranked_probabilities) > 1 else 0.0,
        "rho_trace": float(np.real(np.trace(rho))),
        "rho_purity": float(np.real(np.trace(rho @ rho))),
        "eigenvalues_desc": eigvals_desc.tolist(),
        "effective_rank": int(np.sum(eigvals_desc > float(ev_tol))),
        "S_norm_fro": s_norm,
        "T_norm_fro": t_norm,
        "R_norm_fro": r_norm,
        "S_trace": float(np.real(np.trace(S))),
        "T_trace": float(np.real(np.trace(T))),
        "R_trace": float(np.real(np.trace(R))),
        "reconstruction_error_fro": reconstruction_error,
        "nonempty_bucket_count": nonempty_bucket_count,
        "ledger": ledger,
    }


def summarize_str_partition(
    rho: np.ndarray,
    sorted_qubits: list[cirq.GridQubit],
    *,
    n_classes: int,
    ev_tol: float = 1e-8,
    cmp_tol: float = 1e-9,
) -> dict[str, object]:
    measurement_ops = [
        projector_last_readout_computational(
            outcome_index,
            sorted_qubits,
            n_classes=int(n_classes),
        )
        for outcome_index in range(int(n_classes))
    ]
    return summarize_str_partition_from_measurements(
        rho,
        measurement_ops,
        labels=projector_labels(int(n_classes)),
        ev_tol=float(ev_tol),
        cmp_tol=float(cmp_tol),
    )
