import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse
from typing import Text

# Set the default model name
DEFAULT_MODEL_NAME = "meta-llama/Llama-2-7b-hf"

def download_model(model_name: Text):
    """
    Downloads the specified model and tokenizer to the local Hugging Face cache.
    """
    print(f"--- Starting download for model: {model_name} ---")

    # -----------------------------
    # 1. Download Tokenizer
    # -----------------------------
    try:
        print(f"1. Downloading tokenizer for {model_name}...")
        # The from_pretrained call automatically handles downloading and caching
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"   Tokenizer download complete. Cached at: {tokenizer.name_or_path}")
    except Exception as e:
        print(f"ERROR: Tokenizer download failed. Please check network or model name. Error: {e}")
        return

    # -----------------------------
    # 2. Download Model
    # -----------------------------
    # Loading the model structure and configuration automatically triggers the weights download to the local cache.
    try:
        print(f"2. Downloading model weights for {model_name}...")
        
        # Load the model with standard settings (no 4-bit quantization here) 
        # to ensure the full weights are downloaded and saved to the cache.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True # Use low CPU memory during the initial loading phase
        )
        print(f"   Model weights download complete.")
        
    except Exception as e:
        print(f"ERROR: Model download failed. Ensure you are logged into Hugging Face and have access rights. Error: {e}")
        return

    print("\n--- Model and tokenizer successfully downloaded to local cache. ---")

def main():
    parser = argparse.ArgumentParser(description="Download Llama-2 model to local cache.")
    parser.add_argument('--model_name', 
                        type=str, 
                        default=DEFAULT_MODEL_NAME,
                        help='Hugging Face model ID to download.')
    args = parser.parse_args()
    
    # Check if login might be required (Llama models often are gated)
    if 'LLAMA' in args.model_name.upper():
        print("NOTE: Downloading Llama models usually requires logging into Hugging Face and having access to the model.")
        print("Please ensure you have run 'huggingface-cli login' or set the HUGGINGFACE_TOKEN environment variable.")
        
    download_model(args.model_name)

if __name__ == "__main__":
    main()