#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_ID="${CUDA_ID:-0}"
NUM_TRIALS="${NUM_TRIALS:-5}"
NUM_EPOCHS_3M="${NUM_EPOCHS_3M:-1}"
NUM_EPOCHS_6M="${NUM_EPOCHS_6M:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-5e-4}"
SEED_UPPER="${SEED_UPPER:-1000000}"
MODELS="${MODELS:-OCT2Former}"
DATASETS="${DATASETS:-3M 6M}"

if (( NUM_TRIALS < 1 )); then
  echo "NUM_TRIALS must be at least 1." >&2
  exit 1
fi

if (( SEED_UPPER < 1 )); then
  echo "SEED_UPPER must be at least 1." >&2
  exit 1
fi

read -r -a models <<< "${MODELS}"
read -r -a datasets <<< "${DATASETS}"

random_seed() {
  "${PYTHON_BIN}" - "${SEED_UPPER}" <<'PY'
import secrets
import sys

print(secrets.randbelow(int(sys.argv[1])))
PY
}

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    case "${dataset}" in
      3M)
        dataset_name="OCTA-3M"
        num_epochs="${NUM_EPOCHS_3M}"
        dataset_file_list="utils/OCTA_3M.csv"
        data_root="OCTA_3M/ProjectionMaps/OCTA(ILM_OPL)"
        data_root_aux="OCTA_3M/ProjectionMaps/OCT(ILM_OPL)"
        target_root="OCTA_3M/GT_LargeVessel"
        run_dir="3M"
        ;;
      6M)
        dataset_name="OCTA-6M"
        num_epochs="${NUM_EPOCHS_6M}"
        dataset_file_list="utils/OCTA_6M.csv"
        data_root="OCTA_6M/ProjectionMaps/OCTA(ILM_OPL)"
        data_root_aux="OCTA_6M/ProjectionMaps/OCT(ILM_OPL)"
        target_root="OCTA_6M/GT_LargeVessel"
        run_dir="6M"
        ;;
      *)
        echo "Unsupported dataset: ${dataset}. Use DATASETS='3M', '6M', or '3M 6M'." >&2
        exit 1
        ;;
    esac

    for (( trial = 1; trial <= NUM_TRIALS; trial++ )); do
      seed="$(random_seed)"
      note="${model}_OCTA${dataset}_trial${trial}_seed${seed}"

      echo "================================================================"
      echo "Model=${model} Dataset=${dataset_name} Trial=${trial}/${NUM_TRIALS} Seed=${seed}"
      echo "================================================================"

      TRAIN_SEED="${seed}" "${PYTHON_BIN}" train.py \
        --dataset="${dataset_name}" \
        --network="${model}" \
        --num_epochs="${num_epochs}" \
        --dataset_file_list="${dataset_file_list}" \
        --data_root="${data_root}" \
        --data_root_aux="${data_root_aux}" \
        --target_root="${target_root}" \
        --run_dir="${run_dir}" \
        --in_channel=2 \
        --batch_size="${BATCH_SIZE}" \
        --lr="${LR}" \
        --img_aug \
        --cuda_id="${CUDA_ID}" \
        --note="${note}"
    done
  done
done

echo "All random training trials finished."
