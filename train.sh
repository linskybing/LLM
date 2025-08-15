#! /bin/bash
#SBATCH -A ACD110018
#SBATCH -p gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
#SBATCH -J train
#SBATCH -o train.out

module purge
module load git/2.44.0 cmake
module load nvhpc-24.11_hpcx-2.20_cuda-12.6
spack load gcc@13.4.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ds

export CUDA_HOME="/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/"
export CC=gcc
export CXX=g++

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.7,expandable_segments:True"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=$MASTER_PORT"

TRAIN_BATCH_SIZE=2

deepspeed --master_port $MASTER_PORT pretrain.py \
    --deepspeed_config zero_2.json \
    --batch_size $TRAIN_BATCH_SIZE \
    --seq_len 350 \
    --total_steps 100 
