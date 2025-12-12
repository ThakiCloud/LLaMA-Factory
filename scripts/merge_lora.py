#!/usr/bin/env python3
"""
LoRA Merge Script - Merges LoRA adapter weights into a base model.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter weights into a base model")
    parser.add_argument("-b", "--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("-l", "--lora_path", type=str, required=True, help="Path to the LoRA adapter checkpoint")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to save the merged model")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="cpu", choices=["cpu", "auto", "cuda"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--save_tokenizer", action="store_true", default=True)
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_str]


def print_step(message: str, quiet: bool = False):
    if not quiet:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")


def merge_lora(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    dtype: str = "bfloat16",
    device_map: str = "cpu",
    trust_remote_code: bool = True,
    save_tokenizer: bool = True,
    quiet: bool = False,
) -> None:
    start_time = time.time()
    
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    torch_dtype = get_torch_dtype(dtype)
    
    print_step(f"Loading base model from {base_model_path}...", quiet)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    print_step("Base model loaded successfully", quiet)
    
    print_step(f"Loading LoRA adapter from {lora_path}...", quiet)
    model = PeftModel.from_pretrained(base_model, lora_path)
    print_step("LoRA adapter loaded successfully", quiet)
    
    print_step("Merging LoRA weights into base model...", quiet)
    merged_model = model.merge_and_unload()
    print_step("Merge completed successfully", quiet)
    
    print_step(f"Saving merged model to {output_path}...", quiet)
    merged_model.save_pretrained(output_path)
    print_step("Merged model saved successfully", quiet)
    
    if save_tokenizer:
        print_step("Saving tokenizer...", quiet)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
        tokenizer.save_pretrained(output_path)
        print_step("Tokenizer saved successfully", quiet)
    
    elapsed_time = time.time() - start_time
    print_step(f"✅ Merge completed in {elapsed_time:.2f} seconds", quiet)
    print_step(f"📁 Merged model saved to: {output_path}", quiet)


def main():
    args = parse_args()
    try:
        merge_lora(
            base_model_path=args.base_model,
            lora_path=args.lora_path,
            output_path=args.output_path,
            dtype=args.dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            save_tokenizer=args.save_tokenizer,
            quiet=args.quiet,
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during merge: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
