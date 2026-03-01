"""Training configuration for QLoRA SFT and GRPO."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# === Model ===
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
# Alternative: "mistralai/Ministral-8B-Instruct-2410"

# === Paths ===
TRAINING_DATA = PROJECT_ROOT / "data" / "training_data.jsonl"
SFT_OUTPUT_DIR = PROJECT_ROOT / "models" / "sft-lora"
GRPO_OUTPUT_DIR = PROJECT_ROOT / "models" / "grpo-lora"
MERGED_OUTPUT_DIR = PROJECT_ROOT / "models" / "merged"

# === QLoRA Config ===
QLORA_CONFIG = {
    "r": 64,                    # LoRA rank
    "lora_alpha": 16,           # LoRA alpha
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# === SFT Training Config ===
SFT_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,  # Effective batch size = 16
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "max_seq_length": 2048,
    "fp16": False,
    "bf16": True,               # RTX 5080 supports bf16
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 3,
    "optim": "paged_adamw_8bit",
    "gradient_checkpointing": True,
    "report_to": "wandb",
}

# === GRPO Config ===
GRPO_CONFIG = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-6,       # Lower LR for RL
    "max_seq_length": 2048,
    "num_generations": 4,        # Generate 4 completions per prompt
    "beta": 0.1,                 # KL penalty coefficient
    "temperature": 0.8,
    "report_to": "wandb",
}

# === W&B Config ===
WANDB_PROJECT = "roblox-luau-codegen"
WANDB_ENTITY = "carpediemhari-n-a"
SFT_RUN_NAME = "sft-mistral-7b-luau"
GRPO_RUN_NAME = "grpo-mistral-7b-luau"

# === HuggingFace Config ===
HF_REPO_ID = "carpediemhari/roblox-luau-mistral-7b"
