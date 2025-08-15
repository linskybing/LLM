import argparse
import deepspeed
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import nltk
import time

nltk.download('words')
from nltk.corpus import words
word_list = words.words()

def generate_random_text(tokenizer, target_token_length=350, max_trials=10):
    for _ in range(max_trials):
        approx_word_len = int(target_token_length * 0.75)
        sent_words = random.choices(word_list, k=approx_word_len)
        text = ' '.join(sent_words)
        tokens = tokenizer(text, truncation=True, padding="max_length",
                           max_length=target_token_length, return_tensors="pt")
        length = tokens.input_ids.size(1)
        if length == target_token_length:
            return tokens.input_ids.squeeze(0)
    return tokens.input_ids.squeeze(0)

def generate_batch(tokenizer, batch_size, seq_len, device):
    return torch.stack([generate_random_text(tokenizer, seq_len) for _ in range(batch_size)]).to(device)

def train_step(ds_engine, inputs, labels):
    outputs = ds_engine(inputs, labels=labels)
    loss = outputs.loss

    ds_engine.backward(loss)
    ds_engine.step()

    return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=350)
    parser.add_argument('--total_steps', type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1, help='DeepSpeed local rank')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model and tokenizer: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.gradient_checkpointing_enable()

    torch.cuda.empty_cache()
    ds_engine, optimizer, _, _ = deepspeed.initialize(model=model, config=args.deepspeed_config if hasattr(args, 'deepspeed_config') else None)

    model.train()

    start_time = time.time()

    total_tokens = 0.0
    total_steps_time = 0.0
    total_mem = 0.0
    max_mem = 0.0

    for step in range(args.total_steps):
        inputs = generate_batch(tokenizer, args.batch_size, args.seq_len, device)
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
        print(f"[Step {step}/{args.total_steps}] Loss: {loss_val:.4f} | Tokens/s: {tokens_per_sec:.2f} | GPU Mem (GB): {mem_gb:.2f}")

    avg_tokens_per_sec = total_tokens / total_steps_time
    total_time = time.time() - start_time

    print(f"Training done in {total_time:.2f} seconds.")
    print(f"Avg Tokens/s: {avg_tokens_per_sec:.2f}")
    print(f"Peak GPU Mem (GB): {torch.cuda.max_memory_allocated() / 1024**3 :.2f}")

if __name__ == "__main__":
    main()
