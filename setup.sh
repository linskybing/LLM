#! /bin/bash

module purge
module load git/2.44.0
module load cuda/11.7
conda create -n deepspeed python=3.12 -y
conda activate deepspeed

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install deepspeed-kernels
DS_BUILD_FUSED_ADAM=1 TORCH_CUDA_ARCH_LIST="7.0" pip install --no-cache-dir --force-reinstall --global-option="build_ext" --global-option="-j6" .

# git clone https://github.com/deepspeedai/DeepSpeed/tree/master

pip install transformers datasets tokenizers
pip install numpy tqdm

deepspeed --num_gpus=1 train_llama2_pretrain.py \
  --batch_size 1 --seq_len 350 --total_steps 100 --deepspeed_config ds_config.json