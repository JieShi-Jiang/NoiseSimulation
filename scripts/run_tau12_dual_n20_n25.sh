#!/usr/bin/env bash
# Parallel tau1/tau2: 20 qubits / 4 outcomes + 25 qubits / 2 outcomes (QCNN).
# Noise strengths (Python literals): 10e-2, 10e-3, 5*10e-2, 10e-1  ->  0.1 0.01 0.5 1.0
# Uses 2 processes × THREADS_PER_JOB (default 128) ≈ 256 hardware threads.
set -euo pipefail

REPO="${REPO:-/home/zhengyangz/work/NoiseSimulation}"
THREADS_PER_JOB="${THREADS_PER_JOB:-128}"
cd "$REPO"
mkdir -p "$REPO/logs"

STRENGTHS=(0.1 0.01 0.5 1.0)

run_bg() {
  local n="$1" c="$2" tag="$3"
  local log="$REPO/logs/tau12_${tag}_$(date +%Y%m%d_%H%M%S).log"
  echo "[launch] n=${n} nclasses=${c} threads=${THREADS_PER_JOB} log=${log}"
  PYTHONUNBUFFERED=1 python -m large_n_certification.run_noisy_qcnn_tau123_scalability \
    --nqubits "$n" \
    --nclasses "$c" \
    --circuit qcnn \
    --noise-strengths "${STRENGTHS[@]}" \
    --threads "$THREADS_PER_JOB" \
    --run-tag "${tag}" \
    >>"$log" 2>&1 &
  echo $!
}

run_bg 20 4 "n20_c4_qcnn_dual"
run_bg 25 2 "n25_c2_qcnn_dual"
wait
echo "[done] both jobs finished"
