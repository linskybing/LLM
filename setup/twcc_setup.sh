#! /bin/bash

set -e

module purge
module load miniconda3/conda24.5.0_py3.9 cmake
module load nvhpc-24.11_hpcx-2.20_cuda-12.6

unset CC
unset CXX

conda create -n ds python=3.12 -y
conda activate ds

conda install -c conda-forge gxx_linux-64=12 -y

export CUDA_HOME="/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$CUDA_HOME/lib64":"/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/targets/x86_64-linux/lib"
export LDFLAGS="-L/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/targets/x86_64-linux/lib"

export CC="x86_64-conda-linux-gnu-gcc"
export CXX="x86_64-conda-linux-gnu-g++"

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install deepspeed-kernels
pip install transformers datasets tokenizers
pip install numpy tqdm nltk
pip install accelerate bitsandbytes mpi4py


DS_BUILD_FUSED_ADAM=1 DS_BUILD_CPU_ADAM=1 TORCH_CUDA_ARCH_LIST="7.0" pip install deepspeed==0.17.4

# # LOGIN
# huggingface-cli login