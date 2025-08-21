#! /bin/bash
#SBATCH -A ACD110018
#SBATCH -p gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH -J train
#SBATCH -o /work/u8644434/LLM/logs/quantization.out

module purge
module load git/2.44.0 cmake
module load nvhpc-24.11_hpcx-2.20_cuda-12.6 cuda/12.8
spack load gcc@13.4.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepspeed

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.7,expandable_segments:True"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

export ROOT="/work/u8644434/LLM"
export LOGS=$ROOT/logs
export CONFIG=$ROOT/configs
export RUN=$ROOT/run

# deepspeed --num_gpus=8 --launcher=openmpi $RUN/quantize.py \
#     --deepspeed_config $CONFIG/quantization.json \
#     --batch_size 1 \
#     --seq_len 350 \
#     --total_steps 100 

mpirun -np 8 bash -c 'python $RUN/quantize.py \
    --deepspeed_config $CONFIG/quantization.json \
    --batch_size 1 \
    --seq_len 350 \
    --total_steps 100'