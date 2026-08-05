#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_ID="${CUDA_ID:-0}"
NUM_TRIALS="${NUM_TRIALS:-5}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-5e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED_UPPER="${SEED_UPPER:-1000000}"
DATA_ROOT="${DATA_ROOT:-/data/OCTA-FRNet-main/dataset/ROSSA}"
RUN_DIR="${RUN_DIR:-ROSSA}"
MODE="${MODE:-train_test}"

if (( NUM_TRIALS < 1 )); then
  echo "NUM_TRIALS must be at least 1." >&2
  exit 1
fi

if (( SEED_UPPER < NUM_TRIALS )); then
  echo "SEED_UPPER must be at least NUM_TRIALS to generate unique seeds." >&2
  exit 1
fi

for split in train_manual train_sam val test; do
  if [[ ! -d "${DATA_ROOT}/${split}/image" || ! -d "${DATA_ROOT}/${split}/label" ]]; then
    echo "Missing ROSSA split directory: ${DATA_ROOT}/${split}/{image,label}" >&2
    exit 1
  fi
done

cd "${SCRIPT_DIR}"

declare -A used_seeds=()

random_seed() {
  local candidate

  while true; do
    candidate="$("${PYTHON_BIN}" - "${SEED_UPPER}" <<'PY'
import secrets
import sys

print(secrets.randbelow(int(sys.argv[1])))
PY
)"
    if [[ -z "${used_seeds[${candidate}]+x}" ]]; then
      used_seeds["${candidate}"]=1
      RANDOM_SEED="${candidate}"
      return
    fi
  done
}

for (( trial = 1; trial <= NUM_TRIALS; trial++ )); do
  random_seed
  seed="${RANDOM_SEED}"
  note="OCT2Former_ROSSA_trial${trial}_seed${seed}"

  echo "================================================================"
  echo "Model=OCT2Former Dataset=ROSSA Trial=${trial}/${NUM_TRIALS} Seed=${seed}"
  echo "================================================================"

  TRAIN_SEED="${seed}" "${PYTHON_BIN}" train.py \
    --dataset=ROSSA \
    --network=OCT2Former \
    --mode="${MODE}" \
    --num_epochs="${NUM_EPOCHS}" \
    --data_root="${DATA_ROOT}" \
    --target_root="${DATA_ROOT}" \
    --run_dir="${RUN_DIR}" \
    --in_channel=1 \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --lr="${LR}" \
    --img_aug \
    --cuda_id="${CUDA_ID}" \
    --note="${note}"
done

echo "All ${NUM_TRIALS} ROSSA training trials finished."