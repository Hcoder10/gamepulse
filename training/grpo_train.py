"""GRPO fine-tuning using our scorers as the reward function.

Usage:
    python training/grpo_train.py [--base-model PATH] [--sft-adapter PATH]

Group Relative Policy Optimization — generates multiple completions per prompt,
scores them with our 4 deterministic scorers, and uses relative ranking as reward.
This is our key differentiator: the evaluation scorers become the training signal.
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import wandb
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig

from training.config import (
    BASE_MODEL,
    TRAINING_DATA,
    SFT_OUTPUT_DIR,
    GRPO_OUTPUT_DIR,
    GRPO_CONFIG,
    WANDB_PROJECT,
    WANDB_ENTITY,
    GRPO_RUN_NAME,
)
from training.reward import compute_reward_batch


def load_prompts() -> Dataset:
    """Load just the prompts (user messages) from training data for GRPO."""
    prompts = []
    with open(TRAINING_DATA, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            messages = sample["messages"]
            # Extract user message as the prompt
            for msg in messages:
                if msg["role"] == "user":
                    prompts.append({"prompt": msg["content"]})
                    break

    dataset = Dataset.from_list(prompts)
    print(f"Loaded {len(dataset)} prompts for GRPO")
    return dataset


def reward_function(completions: list[str], **kwargs) -> list[float]:
    """Reward function using our scorers.

    Called by GRPOTrainer for each batch of completions.
    Returns list of scalar rewards.
    """
    return compute_reward_batch(completions)


def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning with scorer rewards")
    parser.add_argument("--base-model", type=str, default=None, help="Base model path")
    parser.add_argument("--sft-adapter", type=str, default=None, help="SFT LoRA adapter path")
    parser.add_argument("--max-prompts", type=int, default=200, help="Max prompts to use")
    args = parser.parse_args()

    model_name = args.base_model or BASE_MODEL
    sft_path = args.sft_adapter or str(SFT_OUTPUT_DIR / "final")

    # Check SFT adapter exists
    if not Path(sft_path).exists():
        print(f"WARNING: SFT adapter not found at {sft_path}")
        print("Running GRPO on base model without SFT initialization.")
        sft_path = None

    # Init W&B
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=GRPO_RUN_NAME,
        config={
            "base_model": model_name,
            "sft_adapter": sft_path,
            "method": "GRPO",
            "grpo_config": GRPO_CONFIG,
        },
        tags=["grpo", "rl", "luau", "roblox"],
    )

    # Load prompts
    print("\n=== Loading Prompts ===")
    dataset = load_prompts()
    if args.max_prompts and len(dataset) > args.max_prompts:
        dataset = dataset.select(range(args.max_prompts))
        print(f"Using {len(dataset)} prompts")

    # Load model
    print(f"\n=== Loading Model: {model_name} ===")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left padding for generation

    # Load SFT adapter if available
    if sft_path:
        print(f"Loading SFT adapter from {sft_path}")
        model = PeftModel.from_pretrained(model, sft_path)
        model = model.merge_and_unload()

    # GRPO output dir
    GRPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=str(GRPO_OUTPUT_DIR),
        num_train_epochs=GRPO_CONFIG["num_train_epochs"],
        per_device_train_batch_size=GRPO_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=GRPO_CONFIG["gradient_accumulation_steps"],
        learning_rate=GRPO_CONFIG["learning_rate"],
        max_completion_length=GRPO_CONFIG["max_seq_length"],
        num_generations=GRPO_CONFIG["num_generations"],
        beta=GRPO_CONFIG["beta"],
        temperature=GRPO_CONFIG["temperature"],
        report_to="wandb",
        run_name=GRPO_RUN_NAME,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
    )

    # New LoRA config for GRPO (separate from SFT)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Create GRPO trainer
    print("\n=== Starting GRPO Training ===")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_function,
        train_dataset=dataset,
        args=grpo_config,
        peft_config=lora_config,
    )

    trainer.train()

    # Save
    print("\n=== Saving GRPO Model ===")
    trainer.save_model(str(GRPO_OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(GRPO_OUTPUT_DIR / "final"))

    # Log as artifact
    artifact = wandb.Artifact(
        name="grpo-lora-adapter",
        type="model",
        description="GRPO LoRA adapter trained with scorer rewards",
    )
    artifact.add_dir(str(GRPO_OUTPUT_DIR / "final"))
    wandb.log_artifact(artifact)

    wandb.finish()
    print(f"\nGRPO training complete!")
    print(f"  Model saved: {GRPO_OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
