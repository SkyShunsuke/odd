#!/bin/bash
# Evaluate a trained checkpoint with the K-step sweep (ablation_steps in the
# config): writes eval_results_<K>.csv per K. Run from the repo root.

export CUDA_VISIBLE_DEVICES=0  # REPLACE with the GPU IDs you want to use
GPU_NUM=1      # REPLACE with the number of GPUs you want to use
PORT=12363     # REPLACE with an available port number
CONFIG_FILE=./configs/eval/mvtec_dit_fm_steps.yaml

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPU_NUM \
    --master_port=$PORT \
    main.py --config_file $CONFIG_FILE --task eval
