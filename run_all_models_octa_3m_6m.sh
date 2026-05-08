#!/usr/bin/env bash
set -euo pipefail

CUDA_ID="${CUDA_ID:-0}"
NUM_EPOCHS_3M="${NUM_EPOCHS_3M:-100}"
NUM_EPOCHS_6M="${NUM_EPOCHS_6M:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-5e-4}"

models=("TransUNet" "swinunet" "OCT2Former")

for model in "${models[@]}"; do
  echo "==================== ${model} on OCTA-3M ===================="
  python train.py \
    --dataset='OCTA-3M' \
    --network="${model}" \
    --num_epochs="${NUM_EPOCHS_3M}" \
    --dataset_file_list='utils/OCTA_3M.csv' \
    --data_root='OCTA_3M/Projection Maps/OCTA(ILM_OPL)' \
    --data_root_aux='OCTA_3M/Projection Maps/OCT(ILM_OPL)' \
    --target_root='OCTA_3M/GroundTruth' \
    --run_dir='3M' \
    --in_channel=2 \
    --batch_size="${BATCH_SIZE}" \
    --lr="${LR}" \
    --img_aug \
    --cuda_id="${CUDA_ID}" \
    --note="${model}_OCTA3M"

  echo "==================== ${model} on OCTA-6M ===================="
  python train.py \
    --dataset='OCTA-6M' \
    --network="${model}" \
    --num_epochs="${NUM_EPOCHS_6M}" \
    --dataset_file_list='utils/OCTA_6M.csv' \
    --data_root='OCTA_6M/Projection Maps/OCTA(ILM_OPL)' \
    --data_root_aux='OCTA_6M/Projection Maps/OCT(ILM_OPL)' \
    --target_root='OCTA_6M/GroundTruth' \
    --run_dir='6M' \
    --in_channel=2 \
    --batch_size="${BATCH_SIZE}" \
    --lr="${LR}" \
    --img_aug \
    --cuda_id="${CUDA_ID}" \
    --note="${model}_OCTA6M"
done

echo "All trainings finished."
