#!/bin/bash

# LLaDA Text-Only Training Script
# Usage: bash scripts/run_llada_text_only_training.sh
conda activate ...
# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=offline

# Change to the project directory
cd .../MMaDA

# Run the training with accelerate

accelerate launch \
    --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero1.yaml \
    --main_process_port=8888 \
    training/train_llada.py \
    config=configs/llada_pretraining.yaml

    # --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero1.yaml \
