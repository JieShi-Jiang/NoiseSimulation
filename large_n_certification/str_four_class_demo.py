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

from large_n_certification.mixed_state_tools import (
    projector_labels,
    probabilities_last_readout_from_rho,
    summarize_str_partition,
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


def report_block(title: str, rho: np.ndarray, qubits: list[cirq.GridQubit], labels: list[str]) -> None:
    print(title)
    print("—" * len(title))
    y = probabilities_last_readout_from_rho(rho, qubits, n_classes=4)
    summary = summarize_str_partition(rho, qubits, n_classes=4)
    c = int(summary["predicted_index"])
    s = int(summary["competitor_index"])

    eig_desc = np.asarray(summary["eigenvalues_desc"], dtype=float)
    print(f"  eigenvalues (top 6 desc): {eig_desc[:6]}")
    print(f"  numerical rank (>1e-8): {int(summary['effective_rank'])}")

    print("  Class probs y_k = Tr(P_k ρ):")
    for k in range(4):
        print(f"    k={k} ({labels[k]}): {y[k]:.8f}")

    print(f"  global prediction: c={c} ({labels[c]}),  s={s} ({labels[s]})")

    print("  STR (per eigenvector of ρ, bucket by ⟨P_c⟩ vs ⟨P_s⟩ on |φ_ℓ⟩):")
    for row in summary["ledger"]:
        print(
            f"    λ={float(row['eigenvalue']):.6e} → {row['bucket']}:  "
            f"⟨P_c⟩={float(row['pc']):.6f},  ⟨P_s⟩={float(row['ps']):.6f}"
        )

    print(
        f"  ‖S‖_F={float(summary['S_norm_fro']):.8f}  "
        f"‖T‖_F={float(summary['T_norm_fro']):.8f}  "
        f"‖R‖_F={float(summary['R_norm_fro']):.8f}"
    )
    print(f"  ‖S+T+R−ρ‖_F={float(summary['reconstruction_error_fro']):.3e}")
    print(f"  nonempty buckets among {{S,T,R}}: {int(summary['nonempty_bucket_count'])}/3")
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
    labels = projector_labels(4)

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
