#! /bin/bash

module purge
module load rocm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepspeed

export CC=gcc
export CXX=g++
export LD_LIBRARY_PATH=/home/sky/miniconda3/envs/deepspeed/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/home/sky/miniconda3/envs/deepspeed/lib/libomp.so

export ROOT=/home/sky/LLM
export LOGS=$ROOT/logs
export CONFIG=$ROOT/configs
export RUN=$ROOT/run

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.7,expandable_segments:True"

deepspeed $RUN/quantize.py \
    --deepspeed_config $CONFIG/RQ.json \
    --batch_size 1 \
    --seq_len 350 \
    --total_steps 100 \
    > "$LOGS/RQ_amd.log" 2>&1
