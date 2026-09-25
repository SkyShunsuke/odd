#!/bin/bash
# Evaluate the closed-form optimal denoiser (no trained checkpoint needed).
# Single-GPU only. Run from the repo root.

export CUDA_VISIBLE_DEVICES=0
GPU_NUM=1
PORT=12363
CONFIG_FILE=./configs/eval/optimal_denoiser.yaml

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPU_NUM \
    --master_port=$PORT \
    main.py --config_file $CONFIG_FILE --task eval_optimal
