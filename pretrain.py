import argparse
import deepspeed
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import time
import nltk

# NLTK words
nltk.download('words')
from nltk.corpus import words
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
    device = ds_engine.device
    inputs = inputs.to(device)
    labels = labels.to(device)

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
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=350)
    parser.add_argument('--total_steps', type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1, help='DeepSpeed local rank')
    args = parser.parse_args()

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # Model
    # -----------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=None,          # 讓 DeepSpeed 管理 GPU
        torch_dtype=torch.float16 # fp16
    )
    model.gradient_checkpointing_enable()  # 節省顯存
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    # -----------------------------
    # DeepSpeed initialize
    # -----------------------------
    ds_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        config=args.deepspeed_config,
        dist_init_required=True
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

        tokens_this_step = args.batch_size * args.seq_len
        tokens_per_sec = tokens_this_step / step_time
        total_tokens += tokens_this_step
        total_steps_time += step_time

        mem_gb = torch.cuda.memory_allocated() / 1024**3
        max_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[Step {step}/{args.total_steps}] Loss: {loss_val:.4f} | Tokens/s: {tokens_per_sec:.2f} | "
              f"GPU Mem (GB): {mem_gb:.2f} | Peak Mem: {max_mem_gb:.2f}")

    avg_tokens_per_sec = total_tokens / total_steps_time
    total_time = time.time() - start_time

    print(f"Training done in {total_time:.2f} seconds.")
    print(f"Avg Tokens/s: {avg_tokens_per_sec:.2f}")
    print(f"Peak GPU Mem (GB): {torch.cuda.max_memory_allocated() / 1024**3 :.2f}")

if __name__ == "__main__":
    main()
