#!/usr/bin/env python3
"""
演示两类 *混态* ρ 上的 STR 分解（四类投影测量在最后两个量子比特上；无信道 ⇒ M̃_k = P_k）。

**A — QCNN 混合**：两条独立随机 QCNN（不同种子）制备 |ψ_A⟩,|ψ_B⟩，
       ρ = p|ψ_A⟩⟨ψ_A| + (1−p)|ψ_B⟩⟨ψ_B|，p=(3−√5)/4。
       两条非正交纯态混合后秩一般为 2；各自的本征矢量可能对 (c,s) 同类分区，
       STR 仍可能塌缩到一个桶——这不是 bug。

**B — 结构化混态（默认一并打印）**：前 6 比特固定在 |0⟩^{⊗6}，只在最后两比特上使用显式 4×4 密度矩阵
       ρ_last = ½|00⟩⟨00| + ½|11⟩⟨11|。
       本征矢量恰为 |00⟩,|11⟩：相对竞争类别 (c,s)=(|10⟩,|11⟩) 时分别落入 R 与 T，
       因而 ‖S‖、‖T‖、‖R‖ 可同时非零（演示记号）。

Run::

    conda activate qml_gpu
    cd NoiseSimulation
    python -m large_n_certification.str_four_class_demo --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

import cirq
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def projector_labels() -> list[str]:
    return ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]


def qubit_projector(bit: int) -> np.ndarray:
    assert bit in (0, 1)
    if bit == 0:
        return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    return np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)


def projector_last_two_computational(
    k: int,
    sorted_qubits: list[cirq.GridQubit],
) -> np.ndarray:
    assert 0 <= k <= 3
    qa, qb = sorted_qubits[-2], sorted_qubits[-1]
    ba, bb = k // 2, k % 2
    Pa, Pb = qubit_projector(ba), qubit_projector(bb)
    mats = [np.eye(2, dtype=np.complex128) for _ in sorted_qubits]
    mats[sorted_qubits.index(qa)] = Pa
    mats[sorted_qubits.index(qb)] = Pb
    op = mats[0]
    for m in mats[1:]:
        op = np.kron(op, m)
    return op


def probs_last_two_from_rho(rho: np.ndarray, sorted_qubits: list[cirq.GridQubit]) -> np.ndarray:
    return np.array(
        [np.real(np.trace(projector_last_two_computational(k, sorted_qubits) @ rho)) for k in range(4)]
    )


def embed_prefix_then_last_two(
    rho_last_two: np.ndarray,
    sorted_qubits: list[cirq.GridQubit],
) -> np.ndarray:
    """ρ = |0⟩⟨0|^{⊗(n−2)} ⊗ ρ_last_two，trace 归一."""
    n = len(sorted_qubits)
    assert rho_last_two.shape == (4, 4)
    p0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho_prefix = p0
    for _ in range(n - 3):
        rho_prefix = np.kron(rho_prefix, p0)
    return np.kron(rho_prefix, rho_last_two.astype(np.complex128))


def simulate_qcnn_pure(
    qubits: list[cirq.GridQubit],
    seed: int,
    full_size: int,
) -> np.ndarray:
    np.random.seed(seed)
    from large_n_certification.circuits.qcnn_random import create_model_circuit, random_variables

    circuit = create_model_circuit(
        sorted(qubits),
        random_variables(),
        p=0.0,
        full_size=full_size,
    )
    sim = cirq.Simulator()
    qo = sorted(circuit.all_qubits())
    psi = sim.simulate(circuit, qubit_order=qo).final_state_vector
    return np.asarray(psi).flatten()


def str_partition_mixed(
    rho: np.ndarray,
    sorted_qubits: list[cirq.GridQubit],
    c: int,
    s: int,
    *,
    ev_tol: float = 1e-8,
    cmp_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, str, float, float]]]:
    dim = rho.shape[0]
    w, V = np.linalg.eigh(rho)
    idx_desc = np.argsort(np.real(w))[::-1]
    w = np.real(w[idx_desc])
    V = V[:, idx_desc]

    floor = max(ev_tol, np.max(w) * 1e-12)
    Pc = projector_last_two_computational(c, sorted_qubits)
    Ps = projector_last_two_computational(s, sorted_qubits)

    Z = np.zeros_like(rho)
    S, T, R = Z.copy(), Z.copy(), Z.copy()
    ledger: list[tuple[float, str, float, float]] = []

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
        ledger.append((lam, bucket, pc, ps))

    return S, T, R, ledger


def report_block(title: str, rho: np.ndarray, qubits: list[cirq.GridQubit], labels: list[str]) -> None:
    print(title)
    print("—" * len(title))
    y = probs_last_two_from_rho(rho, qubits)
    order = np.argsort(y)[::-1]
    c, s = int(order[0]), int(order[1])

    w_all = np.linalg.eigvalsh(rho)
    rank_eff = int(np.sum(w_all > 1e-8))
    print(f"  eigenvalues (top 6 desc): {np.sort(w_all)[::-1][:6]}")
    print(f"  numerical rank (>1e-8): {rank_eff}")

    print("  Class probs y_k = Tr(P_k ρ):")
    for k in range(4):
        print(f"    k={k} ({labels[k]}): {y[k]:.8f}")

    print(f"  global prediction: c={c} ({labels[c]}),  s={s} ({labels[s]})")

    S, T, R, ledger = str_partition_mixed(rho, qubits, c, s)

    print("  STR (per eigenvector of ρ, bucket by ⟨P_c⟩ vs ⟨P_s⟩ on |φ_ℓ⟩):")
    for lam, bucket, pc, ps in ledger:
        print(f"    λ={lam:.6e} → {bucket}:  ⟨P_c⟩={pc:.6f},  ⟨P_s⟩={ps:.6f}")

    ns, nt, nr = np.linalg.norm(S, "fro"), np.linalg.norm(T, "fro"), np.linalg.norm(R, "fro")
    print(f"  ‖S‖_F={ns:.8f}  ‖T‖_F={nt:.8f}  ‖R‖_F={nr:.8f}")
    err = np.linalg.norm(S + T + R - rho, "fro")
    print(f"  ‖S+T+R−ρ‖_F={err:.3e}")
    nz = sum(1 for x in (ns, nt, nr) if x > 1e-6)
    print(f"  nonempty buckets among {{S,T,R}}: {nz}/3")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed-b-delta", type=int, default=9137)
    ap.add_argument("--full-size", type=int, default=4)
    ap.add_argument(
        "--skip-structured",
        action="store_true",
        help="only run QCNN mixture block",
    )
    args = ap.parse_args()

    n = 8
    qubits = sorted(cirq.GridQubit.rect(1, n))
    labels = projector_labels()

    print("=== Part A — mixed state from two independent QCNN branches ===\n")
    p_mix = float(0.25 * (3.0 - np.sqrt(5.0)))  # (3−√5)/4

    psi_a = simulate_qcnn_pure(qubits, args.seed, args.full_size)
    psi_b = simulate_qcnn_pure(qubits, args.seed + args.seed_b_delta, args.full_size)

    rho_a = np.outer(psi_a, psi_a.conj())
    rho_b = np.outer(psi_b, psi_b.conj())
    rho_mix = p_mix * rho_a + (1.0 - p_mix) * rho_b
    rho_mix = 0.5 * (rho_mix + rho_mix.conj().T)

    print(f"  ρ = p|ψ_A⟩⟨ψ_A| + (1−p)|ψ_B⟩⟨ψ_B|,  p≈{p_mix:.12f}")
    print(f"  seeds: A={args.seed}, B={args.seed + args.seed_b_delta}")
    print(f"  |⟨ψ_A|ψ_B⟩| = {abs(psi_a.conj() @ psi_b):.8f}\n")

    report_block("Statistics & STR", rho_mix, qubits, labels)

    if args.skip_structured:
        return

    print("=== Part B — structured mixed state (guaranteed split across buckets for illustration) ===\n")
    print(
        "  Fixed first n−2 qubits in |0⟩, last-two-qubit state:\n"
        "      ρ_last = ½|00⟩⟨00| + ½|11⟩⟨11|\n"
        "  Prediction uses global winner among four classes; STR compares top class c vs runner-up s.\n"
    )

    rho_last = np.zeros((4, 4), dtype=np.complex128)
    rho_last[0, 0] = 0.5
    rho_last[3, 3] = 0.5
    rho_struct = embed_prefix_then_last_two(rho_last, qubits)
    assert abs(np.trace(rho_struct) - 1.0) < 1e-10

    report_block("Statistics & STR", rho_struct, qubits, labels)


if __name__ == "__main__":
    main()
