"""QLoRA SFT fine-tuning for Roblox Luau code generation.

Usage:
    python training/sft_train.py [--epochs N] [--batch-size N] [--resume-from PATH]

Fine-tunes Mistral-7B-Instruct using QLoRA (4-bit quantization + LoRA adapters)
on our collected Roblox Luau training data. Tracks everything in W&B.
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from training.config import (
    BASE_MODEL,
    TRAINING_DATA,
    SFT_OUTPUT_DIR,
    QLORA_CONFIG,
    SFT_CONFIG,
    WANDB_PROJECT,
    WANDB_ENTITY,
    SFT_RUN_NAME,
)


def load_training_data() -> Dataset:
    """Load JSONL training data into a HuggingFace Dataset."""
    samples = []
    with open(TRAINING_DATA, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} training samples")

    # Convert messages to the format TRL expects
    texts = []
    for sample in samples:
        messages = sample["messages"]
        texts.append({"messages": messages})

    dataset = Dataset.from_list(texts)

    # 90/10 train/eval split
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")
    return split


def setup_model_and_tokenizer(model_name: str):
    """Load base model with 4-bit quantization and tokenizer."""
    print(f"Loading model: {model_name}")

    # 4-bit quantization config
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
        attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare model for QLoRA training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(**QLORA_CONFIG)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT fine-tuning")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_train_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per_device_train_batch_size")
    parser.add_argument("--model", type=str, default=None, help="Override base model")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Check training data exists
    if not TRAINING_DATA.exists():
        print(f"ERROR: Training data not found at {TRAINING_DATA}")
        print("Run: python scripts/pull_training_data.py && python scripts/format_training_data.py")
        sys.exit(1)

    model_name = args.model or BASE_MODEL

    # Init W&B
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=SFT_RUN_NAME,
        config={
            "base_model": model_name,
            "method": "QLoRA SFT",
            "lora_config": QLORA_CONFIG,
            "training_config": SFT_CONFIG,
        },
        tags=["sft", "qlora", "luau", "roblox"],
    )

    # Load data
    print("\n=== Loading Training Data ===")
    dataset_split = load_training_data()

    # Setup model
    print("\n=== Setting Up Model ===")
    model, tokenizer = setup_model_and_tokenizer(model_name)

    # Create output dir
    SFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Training config
    sft_config_overrides = {}
    if args.epochs:
        sft_config_overrides["num_train_epochs"] = args.epochs
    if args.batch_size:
        sft_config_overrides["per_device_train_batch_size"] = args.batch_size

    merged_config = {**SFT_CONFIG, **sft_config_overrides}

    training_args = SFTConfig(
        output_dir=str(SFT_OUTPUT_DIR),
        num_train_epochs=merged_config["num_train_epochs"],
        per_device_train_batch_size=merged_config["per_device_train_batch_size"],
        gradient_accumulation_steps=merged_config["gradient_accumulation_steps"],
        learning_rate=merged_config["learning_rate"],
        warmup_ratio=merged_config["warmup_ratio"],
        lr_scheduler_type=merged_config["lr_scheduler_type"],
        max_seq_length=merged_config["max_seq_length"],
        fp16=merged_config["fp16"],
        bf16=merged_config["bf16"],
        logging_steps=merged_config["logging_steps"],
        save_steps=merged_config["save_steps"],
        save_total_limit=merged_config["save_total_limit"],
        optim=merged_config["optim"],
        gradient_checkpointing=merged_config["gradient_checkpointing"],
        report_to="wandb",
        run_name=SFT_RUN_NAME,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Create trainer
    print("\n=== Starting Training ===")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        args=training_args,
    )

    # Train
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Save final model
    print("\n=== Saving Model ===")
    trainer.save_model(str(SFT_OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(SFT_OUTPUT_DIR / "final"))

    # Log final metrics
    final_metrics = trainer.evaluate()
    wandb.log({"final_eval_loss": final_metrics["eval_loss"]})

    # Save LoRA adapter info as W&B artifact
    artifact = wandb.Artifact(
        name="sft-lora-adapter",
        type="model",
        description="QLoRA SFT adapter for Roblox Luau code generation",
    )
    artifact.add_dir(str(SFT_OUTPUT_DIR / "final"))
    wandb.log_artifact(artifact)

    wandb.finish()

    print(f"\nTraining complete!")
    print(f"  Model saved: {SFT_OUTPUT_DIR / 'final'}")
    print(f"  W&B run: {WANDB_ENTITY}/{WANDB_PROJECT}")


if __name__ == "__main__":
    main()
