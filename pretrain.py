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

def generate_batch(tokenizer, batch_size, seq_len):
    return torch.stack([generate_random_text(tokenizer, seq_len) for _ in range(batch_size)]).cuda()

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
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by DeepSpeed')
    args = parser.parse_args()

    deepspeed.init_distributed()

    print(f"Rank {torch.distributed.get_rank()} / World size {torch.distributed.get_world_size()}")
    print(f"Loading model and tokenizer: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).cuda()

    ds_engine, optimizer, _, _ = deepspeed.initialize(model=model, config=args.deepspeed_config if hasattr(args, 'deepspeed_config') else None)

    model.train()

    inputs = generate_batch(tokenizer, args.batch_size, args.seq_len)
    labels = inputs.clone()

    start_time = time.time()

    for step in range(args.total_steps):
        torch.cuda.synchronize()
        step_start = time.time()

        loss_val = train_step(ds_engine, inputs, labels)

        torch.cuda.synchronize()
        step_time = time.time() - step_start

        tokens_per_sec = (args.batch_size * args.seq_len) / step_time
        mem_gb = torch.cuda.memory_allocated() / 1024**3

        if step % 10 == 0 or step == args.total_steps - 1:
            print(f"[Step {step}/{args.total_steps}] Loss: {loss_val:.4f} | Tokens/s: {tokens_per_sec:.2f} | GPU Mem (GB): {mem_gb:.2f}")

    total_time = time.time() - start_time
    print(f"Training done in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
