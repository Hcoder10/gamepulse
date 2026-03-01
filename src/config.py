import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")

# Model
MISTRAL_MODEL = "mistral-large-latest"

# Weave
WEAVE_PROJECT = "roblox-luau-codegen"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PROMPT_VERSIONS_DIR = PROJECT_ROOT / "configs" / "prompt_versions"
RESULTS_DIR = PROJECT_ROOT / "results"
SCORER_CONFIG_PATH = PROJECT_ROOT / "configs" / "scorer_config.json"

PROMPT_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
