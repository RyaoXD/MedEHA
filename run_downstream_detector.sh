#!/usr/bin/env bash
set -euo pipefail

CKPT="${CKPT:?Set CKPT=/path/to/medeha.ckpt before running}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/detector.py \
  --gpus 1 --dataset object_cxr --data_pct 0.01 --learning_rate 5e-4 --batch_size 8 --ckpt_path "$CKPT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/detector.py \
  --gpus 1 --dataset object_cxr --data_pct 0.1 --learning_rate 5e-4 --ckpt_path "$CKPT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/detector.py \
  --gpus 1 --dataset object_cxr --data_pct 1 --learning_rate 5e-4 --ckpt_path "$CKPT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/detector.py \
  --gpus 1 --dataset rsna --data_pct 0.01 --learning_rate 5e-4 --batch_size 8 --ckpt_path "$CKPT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/detector.py \
  --gpus 1 --dataset rsna --data_pct 0.1 --learning_rate 5e-4 --ckpt_path "$CKPT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/detector.py \
  --gpus 1 --dataset rsna --data_pct 1 --learning_rate 5e-4 --ckpt_path "$CKPT"
