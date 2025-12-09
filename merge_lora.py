#!/usr/bin/env python3
"""
Merge LoRA adapter with base model for evaluation
Usage: python merge_lora.py --lora_path ./models/GPT2-Large-LoRA --output_path ./models/GPT2-Large-Merged
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora(lora_path: str, output_path: str, base_model: str = "gpt2-large"):
    """
    Merge LoRA adapter with base model and save the merged model
    
    Args:
        lora_path: Path to the LoRA adapter checkpoint
        output_path: Path to save the merged model
        base_model: Base model name or path (default: gpt2-large)
    """
    print(f"Loading base model: {base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model)
    
    print(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging LoRA adapter with base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Successfully merged and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for merged model")
    parser.add_argument("--base_model", type=str, default="gpt2-large", help="Base model name or path")
    
    args = parser.parse_args()
    
    merge_lora(args.lora_path, args.output_path, args.base_model)

if __name__ == "__main__":
    main()
