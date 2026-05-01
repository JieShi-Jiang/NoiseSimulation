"""
Large-n certification plan (design hooks).

Goal: support analysis aligned with Certifying Adversarial Robustness / QDP paper
without storing dense 2^n x 2^n objects when n is large.

Tiers
-----
1. **Exact (small n)** — Density-matrix simulation O(4^n) or statevector O(2^n)
   for verifying formulas (Theorem 4.1-style constructions on toy Hilbert spaces).

2. **Noiseless surrogate** — Same QCNN/random geometry but gates unitary-only:
   gives y_k(|psi><psi|) = <psi|M_k|psi> via statevector; cheap upper-bound intuition.

3. **Noisy large-n (scalable targets)** —
   - **Output probabilities** y_k(rho) = Tr(M_k E(rho)): Monte Carlo quantum trajectories
     (Kraus sampling) — memory O(n) per trajectory, cost O(samples * |circuit|).
   - **Spectral quantities on E^dagger(M_k)**: reuse parent `qlipschitz.py`
     (TN + power iteration), not full eigendecomposition.

4. **What stays hard at large n** — Full spectral decomposition of rho into S,T,R
   for Theorem 4.1 Case (1) requires either mixed-state simulation (exponential)
   or a surrogate (low-rank / sampled eigenvectors); not implemented as exact here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LargeNCertificationPlan:
    """Serializable summary of recommended pipelines."""

    max_exact_density_matrix_qubits: int = 12
    """Above this, avoid ``DensityMatrixSimulator`` for full rho."""

    recommended_statevector_probe_qubits: int = 28
    """Heuristic: noiseless statevector still feasible on large workstations."""

    trajectory_sampling_default_shots: int = 2000
    """Monte Carlo Kraus trajectories for noisy Born probabilities."""

    lipschitz_reference: str = "../qlipschitz.py"


def describe_limits() -> str:
    return __doc__ or ""


Mode = Literal["exact_dm", "noiseless_statevector", "noisy_trajectories", "lipschitz_spectral"]
