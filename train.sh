#! /bin/bash
#SBATCH -A ACD110018
#SBATCH -p gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -J train
#SBATCH -o train.out

module purge
module load git/2.44.0 gcc10/10.2.1
module load cuda/12.8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama2

export TORCH_CUDA_ARCH_LIST="7.0"
export CC=gcc
export CXX=g++

deepspeed --num_gpus=1 pretrain.py \
  --batch_size 1 --seq_len 350 --total_steps 100 --deepspeed_config native.json