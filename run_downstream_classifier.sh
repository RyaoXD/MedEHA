#!/usr/bin/env bash
# Linear-probe classification. Set CKPT to your pretrained MedEHA Lightning checkpoint.
set -euo pipefail

CKPT="${CKPT:?Set CKPT=/path/to/medeha.ckpt before running}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/classifier.py \
  --gpus 1 --data_pct 0.01 --dataset chexpert --learning_rate 5e-4 --ckpt_path "$CKPT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/classifier.py \
  --gpus 1 --data_pct 0.1 --dataset chexpert --learning_rate 5e-4 --ckpt_path "$CKPT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/classifier.py \
  --gpus 1 --data_pct 1 --dataset chexpert --learning_rate 5e-4 --ckpt_path "$CKPT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/classifier.py \
  --gpus 1 --data_pct 0.01 --dataset rsna --batch_size 16 --learning_rate 5e-4 --ckpt_path "$CKPT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/classifier.py \
  --gpus 1 --data_pct 0.1 --dataset rsna --batch_size 16 --learning_rate 5e-4 --ckpt_path "$CKPT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/classifier.py \
  --gpus 1 --data_pct 1 --dataset rsna --batch_size 16 --learning_rate 5e-4 --ckpt_path "$CKPT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/classifier.py \
  --gpus 1 --data_pct 0.01 --dataset covidx --batch_size 16 --learning_rate 5e-4 --ckpt_path "$CKPT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/classifier.py \
  --gpus 1 --data_pct 0.1 --dataset covidx --batch_size 16 --learning_rate 5e-4 --ckpt_path "$CKPT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./medeha/models/downstream/classifier.py \
  --gpus 1 --data_pct 1 --dataset covidx --batch_size 16 --learning_rate 5e-4 --ckpt_path "$CKPT"
