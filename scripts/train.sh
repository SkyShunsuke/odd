#!/bin/bash
# Train the flow-matching velocity field on frozen EfficientNet-B4 latents
# (multi-class MVTec AD). Run from the repo root; `mkdir -p logs` first.

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # REPLACE with the GPU IDs you want to use
GPU_NUM=8      # REPLACE with the number of GPUs you want to use
PORT=12363     # REPLACE with an available port number
CONFIG_FILE=./configs/train/mvtec_dit_fm.yaml

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPU_NUM \
    --master_port=$PORT \
    main.py --config_file $CONFIG_FILE --task train
