#!/usr/bin/env bash
# Paper pre-training command for MedEHA.
# Losses in code: GA = L_GA, ICA = L_PSA, LGA = L_LGA, PFA = L_PFA.
# Do not use --experiment_name test: that name swaps in a debug configuration.
set -euo pipefail

export MEDEHA_DATA_DIR="${MEDEHA_DATA_DIR:-/mnt/E/dataset}"
export MEDEHA_OUTPUT_DIR="${MEDEHA_OUTPUT_DIR:-/mnt/C/kxm}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/pretrain/pretrain.py \
  --gpus 1 \
  --batch_size 72 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --seed 42 \
  --experiment_name medeha_paper \
  --entity_extraction \
  --max_n_entity 30 \
  --with_report_feat \
  --twostep_attention \
  --entity_layer_type decoder \
  --n_entity_layers 4 \
  --report_layer_type decoder \
  --n_report_layers 4 \
  --language_group \
  --group_threshold 0.05 \
  --enable_loss_lga \
  --enable_loss_pfa \
  --enable_loss_ica \
  --instance_prototype random \
  --num_instance_prototypes 500 \
  --freeze_prototypes_epochs 5 \
  --ga_temp 0.05 \
  --ica_temp 0.1 \
  --group_temp 0.2 \
  --pfa_temp 0.2 \
  --lambda_ga 1 \
  --lambda_ica 1 \
  --lambda_group 1 \
  --lambda_pfa 1
