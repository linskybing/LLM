import argparse
import deepspeed
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import random
import time
import nltk
import bitsandbytes as bnb
# NLTK words
nltk.download('words')
from nltk.corpus import words
import os
word_list = words.words()

# -----------------------------
# Random text generation utils
# -----------------------------
def generate_random_text(tokenizer, target_token_length=350, max_trials=10):
    for _ in range(max_trials):
        approx_word_len = int(target_token_length * 0.75)
        sent_words = random.choices(word_list, k=approx_word_len)
        text = ' '.join(sent_words)
        tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=target_token_length,
            return_tensors="pt"
        )
        if tokens.input_ids.size(1) == target_token_length:
            return tokens.input_ids.squeeze(0)
    return tokens.input_ids.squeeze(0)

def generate_batch(tokenizer, batch_size, seq_len, device):
    batch = torch.stack([generate_random_text(tokenizer, seq_len) for _ in range(batch_size)])
    return batch.to(device)

# -----------------------------
# Training step
# -----------------------------
def train_step(ds_engine, inputs, labels):
    outputs = ds_engine(inputs, labels=labels)
    loss = outputs.loss
    ds_engine.backward(loss)
    ds_engine.step()
    return loss.item()

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--model_name', type=str, default='/work/jonathan0hsu/llm-inference/model/Llama-2-7B-hf/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=350)
    parser.add_argument('--total_steps', type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # Distributed init
    # -----------------------------
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    device = torch.device(f"cuda:{local_rank}")

    # -----------------------------
    # Model
    # -----------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": device},
        use_cache = False
    )
    model.gradient_checkpointing_enable()
    model.to(device)
    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    # -----------------------------
    # DeepSpeed initialize
    # -----------------------------
    ds_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        config=args.deepspeed_config,
        dist_init_required=False
    )

    model.train()

    # -----------------------------
    # Training loop
    # -----------------------------
    start_time = time.time()
    total_tokens = 0.0
    total_steps_time = 0.0

    for step in range(args.total_steps):
        inputs = generate_batch(tokenizer, args.batch_size, args.seq_len, ds_engine.device)
        labels = inputs.clone()

        torch.cuda.synchronize()
        step_start = time.time()

        loss_val = train_step(ds_engine, inputs, labels)

        torch.cuda.synchronize()
        step_time = time.time() - step_start

        local_tokens = torch.tensor([args.batch_size * args.seq_len], device=ds_engine.device)
        if dist.is_initialized() and world_size > 1:
            dist.all_reduce(local_tokens, op=dist.ReduceOp.SUM)
        tokens_this_step = local_tokens.item()

        tokens_per_sec = tokens_this_step / step_time
        total_tokens += tokens_this_step
        total_steps_time += step_time

        if rank == 0:
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            max_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[Step {step+1}/{args.total_steps}] Loss: {loss_val:.4f} | "
                  f"Global Tokens/s: {tokens_per_sec:.2f} | GPU Mem (GB): {mem_gb:.2f} | Peak Mem: {max_mem_gb:.2f}")

    if rank == 0:
        avg_tokens_per_sec = total_tokens / total_steps_time
        total_time = time.time() - start_time
        print(f"\nTraining done in {total_time:.2f} seconds.")
        print(f"Avg Global Tokens/s: {avg_tokens_per_sec:.2f}")
        print(f"Peak GPU Mem (GB): {torch.cuda.max_memory_allocated() / 1024**3 :.2f}")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
