# Large-$n$ certification design (NoiseSimulation subpackage)

This folder hosts a **computational plan** for analysing quantum classifiers at qubit counts where dense $2^n\times 2^n$ state manipulation is impossible. Circuit geometry follows **`evaluate_qcnn_model.py`** (`create_model_circuit`, pooling + convolution), with an optional **random Heisenberg-style** brick (`random_heisenberg_circuit`) for cheap experiments.

## Problem split

| Target | Small $n$ | Large $n$ |
|--------|-------------|-----------|
| Output probabilities $y_k(\rho)=\mathrm{Tr}(\mathcal E^\dagger(M_k)\rho)$ | Exact: `DensityMatrixSimulator` or noiseless `Simulator` | Trajectory MC (Kraus sampling), or tensor-network / experimental shots |
| Spectral bounds on $\mathcal E^\dagger(M_k)$ | Dense eigen decomposition | TN + Krylov (`../qlipschitz.py`) |
| Theorem 4.1 $S,T,R$ from $\rho$ | Full `eigh($\rho$)` | Surrogate: low-rank / sampled eigenvectors / defer |

## Modules

- `scheme.py` — tier summary (`LargeNCertificationPlan`).
- `circuits/qcnn_random.py` — QCNN recursion + random layered circuit + noise (same spirit as repo QCNN script).
- `estimation.py` — **noiseless** last-qubit probability via stripped unitaries; **noisy** probability for small $n$ via density matrix.
- `run_noisy_qcnn_tau123_scalability.py` — scalable noisy-ansatz `\tau_1/\tau_2` runner with depolarizing-only `\tau_3`, using TN probability contractions for computational-basis inputs when dense density matrices are too large.
- `demo.py` — runnable smoke test (`python -m large_n_certification.demo` from `NoiseSimulation/`).

## Commands

```bash
conda activate qml_gpu
cd /path/to/NoiseSimulation
python -m large_n_certification.demo --nqubits 8 --circuit qcnn --noise-p 0.01
python -m large_n_certification.demo --nqubits 12 --circuit random --noise-p 0.02
python -m large_n_certification.run_noisy_qcnn_tau123_scalability --nqubits 20 --nclasses 4 --noise-types depolarizing amplitude_damping bit_flip
```

Use **`--max-dm`** only when $n$ is small enough for `DensityMatrixSimulator` (memory $\Theta(4^n)$).

## Next implementation hooks (not shipped)

1. **Trajectory estimator** — repeatedly sample Kraus outcomes along `circuit` moments; average Born estimates for each POVM element (memory $O(n)$ per shot).
2. **MPS/MPDO** — replace statevector/DM simulators when circuits admit low bond dimension.
