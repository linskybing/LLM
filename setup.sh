#! /bin/bash

module purge
module load git/2.44.0 cmake
module load nvhpc-24.11_hpcx-2.20_cuda-12.6 gcc10/10.2.1
conda create -n deepspeed python=3.12 -y
conda activate deepspeed

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install deepspeed-kernels
export CUDA_HOME="/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/11.8/targets/x86_64-linux/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

export CC=gcc
export CXX=g++
DS_BUILD_FUSED_ADAM=1 TORCH_CUDA_ARCH_LIST="7.0" pip install --global-option="build_ext" --global-option="-j6" .
DS_BUILD_OPS=1 TORCH_CUDA_ARCH_LIST="7.0" pip install --global-option="build_ext" --global-option="-j6" .
# git clone https://github.com/deepspeedai/DeepSpeed/tree/master

pip install transformers datasets tokenizers
pip install numpy tqdm nltk

deepspeed --num_gpus=1 train_llama2_pretrain.py \
  --batch_size 1 --seq_len 350 --total_steps 100 --deepspeed_config ds_config.json