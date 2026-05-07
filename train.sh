#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source venv/bin/activate

RUN_NAME="${1:-train_v2}"

yolo task=detect mode=train \
  model=yolov8s.pt \
  data=data.yaml \
  epochs=200 \
  patience=30 \
  imgsz=640 \
  device=mps \
  cos_lr=True \
  lr0=0.005 \
  name="$RUN_NAME"
