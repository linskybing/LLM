#! /bin/bash
#SBATCH -A ACD110018
#SBATCH -p gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
#SBATCH -J train
#SBATCH -o train.out

module purge
module load git/2.44.0 cmake
module load nvhpc-24.11_hpcx-2.20_cuda-12.6

spack load gcc@13.4.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepspeed
export CUDA_HOME="/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/"
export CC=gcc
export CXX=g++

deepspeed --num_gpus=1 --master_port 29501 pretrain.py \
  --batch_size 1 --seq_len 350 --total_steps 100 --deepspeed_config zero_3ORF.json