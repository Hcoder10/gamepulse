#!/usr/bin/env python3
"""
Merge LoRA adapters into the base model and push full models to HuggingFace.

This creates standalone models that can be served by HF Inference Endpoints,
vLLM, TGI, or any inference platform without needing PEFT at runtime.

Requirements:
    pip install transformers peft torch accelerate huggingface_hub

Usage:
    # Merge SFT adapter
    python scripts/merge_and_push.py \
        --adapter squaredcuber/roblox-luau-mistral-7b-2 \
        --output squaredcuber/roblox-luau-mistral-7b-sft-merged \
        --push

    # Merge RFT adapter
    python scripts/merge_and_push.py \
        --adapter squaredcuber/roblox-luau-mistral-7b-rft \
        --output squaredcuber/roblox-luau-mistral-7b-rft-merged \
        --push

    # Both at once
    python scripts/merge_and_push.py --all --push
"""
import argparse
import torch
from pathlib import Path


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

ADAPTERS = {
    "sft": {
        "adapter": "squaredcuber/roblox-luau-mistral-7b-2",
        "output": "squaredcuber/roblox-luau-mistral-7b-sft-merged",
    },
    "rft": {
        "adapter": "squaredcuber/roblox-luau-mistral-7b-rft",
        "output": "squaredcuber/roblox-luau-mistral-7b-rft-merged",
    },
}


def merge_and_push(adapter_id: str, output_id: str, push: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\n{'='*60}")
    print(f"Merging: {adapter_id}")
    print(f"Output:  {output_id}")
    print(f"{'='*60}\n")

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print(f"Loading LoRA adapter: {adapter_id}")
    model = PeftModel.from_pretrained(base_model, adapter_id)

    print("Merging weights...")
    model = model.merge_and_unload()

    local_path = Path("merged_models") / output_id.split("/")[-1]
    print(f"Saving to {local_path}")
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)

    if push:
        print(f"Pushing to HuggingFace: {output_id}")
        model.push_to_hub(output_id, private=False)
        tokenizer.push_to_hub(output_id, private=False)
        print(f"Pushed: https://huggingface.co/{output_id}")

    print("Done!\n")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters and push to HF")
    parser.add_argument("--adapter", help="HF adapter model ID")
    parser.add_argument("--output", help="HF output model ID for merged model")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--all", action="store_true", help="Merge all adapters (SFT + RFT)")
    args = parser.parse_args()

    if args.all:
        for name, cfg in ADAPTERS.items():
            merge_and_push(cfg["adapter"], cfg["output"], push=args.push)
    elif args.adapter and args.output:
        merge_and_push(args.adapter, args.output, push=args.push)
    else:
        parser.error("Provide --adapter and --output, or use --all")


if __name__ == "__main__":
    main()
