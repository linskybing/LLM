#! /bin/bash

module purge
module load git/2.44.0 cmake
module load nvhpc-24.11_hpcx-2.20_cuda-12.6
spack load gcc@13.4.0
conda create -n ds python=3.12 -y
conda activate ds

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install deepspeed-kernels

git clone --recursive https://github.com/NVIDIA/cutlass.git
cd cutlass

export CUDA_HOME="/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/"
export CC=gcc
export CXX=g++

DS_BUILD_FUSED_ADAM=1 DS_BUILD_CPU_ADAM=1 TORCH_CUDA_ARCH_LIST="7.0" pip install --global-option="build_ext" --global-option="-j6" .
DS_BUILD_OPS=1 TORCH_CUDA_ARCH_LIST="7.0" pip install --global-option="build_ext" --global-option="-j6" .
# git clone https://github.com/deepspeedai/DeepSpeed/tree/master

pip install transformers datasets tokenizers
pip install numpy tqdm nltk
pip install accelerate