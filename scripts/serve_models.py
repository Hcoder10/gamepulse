#!/usr/bin/env python3
"""
Serve Mistral-7B base + SFT/RFT LoRA adapters via vLLM.

Starts an OpenAI-compatible API at http://0.0.0.0:8000 that serves:
  - model="base"  → Mistral-7B-Instruct-v0.3 (no adapter)
  - model="sft"   → SFT LoRA adapter (squaredcuber/roblox-luau-mistral-7b-2)
  - model="rft"   → RFT LoRA adapter (squaredcuber/roblox-luau-mistral-7b-rft)

Requirements:
    pip install vllm

Usage (on your H100 or any GPU machine):
    python scripts/serve_models.py
    python scripts/serve_models.py --port 8000 --host 0.0.0.0

Then set MODEL_ENDPOINT_URL=http://<your-ip>:8000 in the webapp .env

To expose to the internet (for Railway), use ngrok:
    ngrok http 8000
    Then set MODEL_ENDPOINT_URL=https://<ngrok-id>.ngrok-free.app
"""
import argparse
import subprocess
import sys


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
SFT_ADAPTER = "squaredcuber/roblox-luau-mistral-7b-2"
RFT_ADAPTER = "squaredcuber/roblox-luau-mistral-7b-rft"


def main():
    parser = argparse.ArgumentParser(description="Serve Luau models with vLLM")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--served-model-name", "base",
        "--enable-lora",
        "--lora-modules",
        f"sft={SFT_ADAPTER}",
        f"rft={RFT_ADAPTER}",
        "--host", args.host,
        "--port", str(args.port),
        "--max-lora-rank", str(args.max_lora_rank),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", "8192",
    ]

    print(f"Starting vLLM server on {args.host}:{args.port}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  SFT adapter: {SFT_ADAPTER}")
    print(f"  RFT adapter: {RFT_ADAPTER}")
    print()
    print("Available models: base, sft, rft")
    print(f"API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
