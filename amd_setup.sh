#! /bin/bash

sudo docker pull deepspeed/rocm501:ds060_pytorch110

sudo docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -v /home/sky/LLM:/workspace \
  --name rocm_deepspeed_sky \
  deepspeed/rocm501:ds060_pytorch110 \
  bash

pip install \
  transformers==4.28.1 \
  datasets==2.12.0 \
  tokenizers==0.13.2 \
  safetensors==0.3.1
pip install numpy tqdm nltk
pip install sentencepiece

deepspeed --num_gpus=1 pretrain_amd.py \
  --batch_size 1 --seq_len 350 --total_steps 100 --deepspeed_config native.json