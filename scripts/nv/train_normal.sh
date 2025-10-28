#! /bin/bash
#SBATCH -A ACD110018
#SBATCH -p gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH -J train
#SBATCH -o /work/u8644434/LLM/logs/nv/zero_3.out

module purge
module load miniconda3/conda24.5.0_py3.9 cmake
module load nvhpc-24.11_hpcx-2.20_cuda-12.6

conda activate /work/u8644434/deepspeed-pretrain

export PYTORCH_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.7,expandable_segments:True"

# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# nvidia-smi
# python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

export ROOT="/work/u8644434/LLM" # [TODO] set your dir location
export LOGS=$ROOT/logs
export CONFIG=$ROOT/configs
export RUN=$ROOT/run

mpirun -np 8 bash -c 'python $RUN/pretrain.py \
    --deepspeed_config $CONFIG/nvidia/zero_3.json \
    --batch_size 1 \
    --seq_len 350 \
    --total_steps 100'
