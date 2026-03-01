"""Run comparative evaluation: base prompted (Mistral API) vs SFT vs RFT (HF Inference API).

Usage:
    python scripts/run_comparison.py

Evaluates three models side-by-side using W&B Weave:
  1. Base Mistral (mistral-large-latest) with a crafted Roblox Luau system prompt
  2. SFT fine-tuned model via HuggingFace Inference API
  3. RFT fine-tuned model via HuggingFace Inference API

All 15 test tasks are scored by 5 scorers (syntax, api, bug, quality, task_completion).
Results are logged to Weave and saved to results/comparison.json.
"""

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import requests
import weave

from src.config import WEAVE_PROJECT, RESULTS_DIR
from src.prompts import DEFAULT_SYSTEM_PROMPT
from src.mistral_client import generate_completion
from data.test_tasks import TEST_TASKS
from scorers import SyntaxScorer, ApiScorer, BugScorer, QualityScorer, TaskCompletionScorer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_SFT_MODEL = "squaredcuber/roblox-luau-mistral-7b-2"
HF_RFT_MODEL = "squaredcuber/roblox-luau-mistral-7b-rft"
HF_API_URL = "https://api-inference.huggingface.co/models/{model_id}"

# Minimal system prompt for the fine-tuned models (they already learned conventions)
HF_SYSTEM_PROMPT = (
    "You are an expert Roblox Luau programmer. "
    "Generate complete, production-ready Luau scripts. "
    "Output only the Luau code. No markdown fences, no explanations."
)

# HF Inference API retry config
HF_MAX_RETRIES = 5
HF_BASE_DELAY = 5.0  # seconds -- cold-start can take a while


# ---------------------------------------------------------------------------
# Weave Model wrappers
# ---------------------------------------------------------------------------

class BaseMistralGenerator(weave.Model):
    """Base prompted model using the Mistral API (mistral-large-latest)."""

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model_name: str = "mistral-large-latest"
    temperature: float = 0.7

    @weave.op()
    def predict(self, task_description: str) -> str:
        raw = generate_completion(
            system_prompt=self.system_prompt,
            user_prompt=task_description,
            model=self.model_name,
            temperature=self.temperature,
        )
        return _strip_markdown_fences(raw)


class HFInferenceGenerator(weave.Model):
    """Generator that calls the HuggingFace Inference API for a hosted model."""

    hf_model_id: str
    model_name: str
    temperature: float = 0.7
    max_new_tokens: int = 2048
    system_prompt: str = HF_SYSTEM_PROMPT

    @weave.op()
    def predict(self, task_description: str) -> str:
        """Send a chat-formatted prompt to the HF Inference API and return generated code."""
        url = HF_API_URL.format(model_id=self.hf_model_id)
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }

        # Build Mistral-Instruct chat template manually.
        # Mistral v0.3 format: <s>[INST] {system}\n\n{user} [/INST]
        prompt_text = (
            f"<s>[INST] {self.system_prompt}\n\n{task_description} [/INST]\n"
        )

        payload = {
            "inputs": prompt_text,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False,
            },
        }

        for attempt in range(HF_MAX_RETRIES):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)

                # Model is loading -- wait and retry
                if resp.status_code == 503:
                    body = resp.json()
                    wait_time = body.get("estimated_time", HF_BASE_DELAY * (2 ** attempt))
                    print(f"    [{self.model_name}] Model loading, waiting {wait_time:.0f}s...")
                    time.sleep(min(wait_time, 120))
                    continue

                # Rate limited
                if resp.status_code == 429:
                    delay = HF_BASE_DELAY * (2 ** attempt)
                    print(f"    [{self.model_name}] Rate limited, waiting {delay:.0f}s...")
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                result = resp.json()

                # The HF Inference API returns a list of dicts with "generated_text"
                if isinstance(result, list) and len(result) > 0:
                    code = result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    code = result.get("generated_text", "")
                else:
                    code = str(result)

                return _strip_markdown_fences(code)

            except requests.exceptions.RequestException as e:
                if attempt == HF_MAX_RETRIES - 1:
                    print(f"    [{self.model_name}] FAILED after {HF_MAX_RETRIES} retries: {e}")
                    return f"-- ERROR: HF Inference API call failed: {e}"
                delay = HF_BASE_DELAY * (2 ** attempt)
                print(f"    [{self.model_name}] Retry {attempt + 1}/{HF_MAX_RETRIES} after {delay:.0f}s: {e}")
                time.sleep(delay)

        return "-- ERROR: HF Inference API exhausted all retries"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(code: str) -> str:
    """Remove markdown code fences if present."""
    code = code.strip()
    code = re.sub(r"^```(?:lua|luau)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def build_dataset() -> weave.Dataset:
    """Build and publish a Weave Dataset from the 15 test tasks."""
    rows = [
        {"task_description": t["task_description"], "category": t["category"]}
        for t in TEST_TASKS
    ]
    dataset = weave.Dataset(name="comparison-tasks-15", rows=rows)
    weave.publish(dataset)
    return dataset


def build_scorers() -> list:
    """Instantiate all 5 scorers."""
    return [
        SyntaxScorer(),
        ApiScorer(),
        BugScorer(),
        QualityScorer(),
        TaskCompletionScorer(),
    ]


def extract_scorer_means(results: dict) -> dict:
    """Extract mean scores from Weave evaluation results.

    Weave returns results keyed by scorer class name, each containing
    the metric dict. We normalize this to a flat dict of score names.
    """
    scores = {}

    # Map from Weave result keys to our display names
    key_map = {
        "SyntaxScorer": "syntax_score",
        "ApiScorer": "api_score",
        "BugScorer": "bug_score",
        "QualityScorer": "quality_score",
        "TaskCompletionScorer": "task_completion_score",
    }

    for scorer_name, metric_key in key_map.items():
        scorer_data = results.get(scorer_name, {})
        if isinstance(scorer_data, dict):
            # Weave nests scores: ScorerName -> metric_key -> mean
            val = scorer_data.get(metric_key, {})
            if isinstance(val, dict):
                scores[metric_key] = val.get("mean", 0.0)
            elif isinstance(val, (int, float)):
                scores[metric_key] = val
            else:
                scores[metric_key] = 0.0
        else:
            scores[metric_key] = 0.0

    # Compute composite average
    score_values = [v for v in scores.values() if isinstance(v, (int, float))]
    scores["composite"] = round(sum(score_values) / max(len(score_values), 1), 4)

    return scores


def print_comparison_table(all_results: dict):
    """Print a formatted comparison table of all model results."""
    metrics = [
        "syntax_score", "api_score", "bug_score",
        "quality_score", "task_completion_score", "composite",
    ]
    metric_labels = {
        "syntax_score": "Syntax",
        "api_score": "API",
        "bug_score": "Bugs",
        "quality_score": "Quality",
        "task_completion_score": "Task Completion",
        "composite": "COMPOSITE",
    }

    # Header
    model_names = list(all_results.keys())
    col_width = 18
    header = f"{'Metric':<20}" + "".join(f"{name:>{col_width}}" for name in model_names)
    separator = "-" * len(header)

    print("\n" + separator)
    print("COMPARATIVE EVALUATION RESULTS")
    print(separator)
    print(header)
    print(separator)

    for metric in metrics:
        label = metric_labels.get(metric, metric)
        row = f"{label:<20}"
        for model_name in model_names:
            scores = all_results[model_name].get("scores", {})
            val = scores.get(metric, 0.0)
            if isinstance(val, (int, float)):
                row += f"{val:>{col_width}.4f}"
            else:
                row += f"{'N/A':>{col_width}}"
        print(row)

    print(separator)

    # Delta row: improvement of SFT and RFT over base
    if "base_prompted" in all_results:
        base_scores = all_results["base_prompted"].get("scores", {})
        print("\nDelta vs base_prompted:")
        for model_name in model_names:
            if model_name == "base_prompted":
                continue
            model_scores = all_results[model_name].get("scores", {})
            base_comp = base_scores.get("composite", 0)
            model_comp = model_scores.get("composite", 0)
            delta = model_comp - base_comp
            sign = "+" if delta >= 0 else ""
            print(f"  {model_name}: {sign}{delta:.4f} composite")
            for metric in metrics[:-1]:  # skip composite
                base_val = base_scores.get(metric, 0)
                model_val = model_scores.get(metric, 0)
                d = model_val - base_val
                s = "+" if d >= 0 else ""
                label = metric_labels.get(metric, metric)
                print(f"    {label}: {s}{d:.4f}")

    print()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

async def run_comparison():
    """Run all three models through evaluation and compare."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Init Weave
    weave.init(WEAVE_PROJECT)

    # Build shared dataset and scorers
    dataset = build_dataset()
    scorers = build_scorers()

    all_results = {}

    # ------------------------------------------------------------------
    # 1. Base Prompted Model (Mistral API)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  MODEL 1/3: Base Prompted (mistral-large-latest via Mistral API)")
    print("=" * 60)

    base_generator = BaseMistralGenerator()
    base_eval = weave.Evaluation(
        dataset=dataset,
        scorers=scorers,
        name="comparison-base-prompted",
    )
    raw_base = await base_eval.evaluate(base_generator)
    base_scores = extract_scorer_means(raw_base)
    all_results["base_prompted"] = {
        "model_id": "mistral-large-latest",
        "model_type": "api_prompted",
        "scores": base_scores,
        "raw_results": {k: str(v)[:500] for k, v in raw_base.items()},
    }
    print(f"  Composite: {base_scores.get('composite', 'N/A')}")

    # ------------------------------------------------------------------
    # 2. SFT Fine-tuned Model (HF Inference API)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  MODEL 2/3: SFT ({HF_SFT_MODEL} via HF Inference API)")
    print("=" * 60)

    if not HF_TOKEN:
        print("  WARNING: HF_TOKEN not set in .env -- skipping HF models")
        print("  Add HF_TOKEN=hf_... to your .env file to enable HF evaluation")
        sft_scores = {"composite": 0.0, "error": "HF_TOKEN not set"}
        all_results["sft"] = {
            "model_id": HF_SFT_MODEL,
            "model_type": "hf_inference_sft",
            "scores": sft_scores,
            "raw_results": {},
        }
    else:
        sft_generator = HFInferenceGenerator(
            hf_model_id=HF_SFT_MODEL,
            model_name="sft-mistral-7b-luau",
        )
        sft_eval = weave.Evaluation(
            dataset=dataset,
            scorers=scorers,
            name="comparison-sft",
        )
        raw_sft = await sft_eval.evaluate(sft_generator)
        sft_scores = extract_scorer_means(raw_sft)
        all_results["sft"] = {
            "model_id": HF_SFT_MODEL,
            "model_type": "hf_inference_sft",
            "scores": sft_scores,
            "raw_results": {k: str(v)[:500] for k, v in raw_sft.items()},
        }
        print(f"  Composite: {sft_scores.get('composite', 'N/A')}")

    # ------------------------------------------------------------------
    # 3. RFT Fine-tuned Model (HF Inference API)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  MODEL 3/3: RFT ({HF_RFT_MODEL} via HF Inference API)")
    print("=" * 60)

    if not HF_TOKEN:
        rft_scores = {"composite": 0.0, "error": "HF_TOKEN not set"}
        all_results["rft"] = {
            "model_id": HF_RFT_MODEL,
            "model_type": "hf_inference_rft",
            "scores": rft_scores,
            "raw_results": {},
        }
    else:
        rft_generator = HFInferenceGenerator(
            hf_model_id=HF_RFT_MODEL,
            model_name="rft-mistral-7b-luau",
        )
        rft_eval = weave.Evaluation(
            dataset=dataset,
            scorers=scorers,
            name="comparison-rft",
        )
        raw_rft = await rft_eval.evaluate(rft_generator)
        rft_scores = extract_scorer_means(raw_rft)
        all_results["rft"] = {
            "model_id": HF_RFT_MODEL,
            "model_type": "hf_inference_rft",
            "scores": rft_scores,
            "raw_results": {k: str(v)[:500] for k, v in raw_rft.items()},
        }
        print(f"  Composite: {rft_scores.get('composite', 'N/A')}")

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print_comparison_table(all_results)

    # ------------------------------------------------------------------
    # Save to JSON
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": timestamp,
        "test_task_count": len(TEST_TASKS),
        "scorers": ["syntax", "api", "bug", "quality", "task_completion"],
        "models": all_results,
    }

    comparison_path = RESULTS_DIR / "comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {comparison_path}")

    # Also save a timestamped copy for history
    history_path = RESULTS_DIR / f"comparison_{timestamp}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"History copy saved to {history_path}")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_comparison())
