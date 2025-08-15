#! /bin/bash

module purge
module load rocm

export CC=clang
export CXX=clang++

conda create -n deepspeed python=3.11 -y
conda activate deepspeed
pip install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip install deepspeed-kernels
pip install triton==3.2.0

# FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
# cd flash-attention &&\
#     git checkout main_perf &&\
#     python setup.py install
DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 pip install --pre --global-option="build_ext" --global-option="-j52" .