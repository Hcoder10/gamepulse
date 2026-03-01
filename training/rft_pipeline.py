"""RFT Pipeline: Rejection Sampling Fine-Tuning with Sonnet Judge.

Single script — run on H100:
    cd /tmp/luau-codegen && git pull && python training/rft_pipeline.py

Generates candidates from SFT model, filters with Claude Sonnet + deterministic scorers,
then retrains on only the best outputs. The model learns from its own best work.
"""

import sys
import gc
import json
import os
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent

# Install deps if needed
import subprocess
subprocess.run(["pip", "install", "-q",
    "anthropic", "transformers", "trl", "peft",
    "datasets", "accelerate", "wandb", "weave",
    "huggingface_hub", "sentencepiece", "protobuf",
    "python-dotenv", "mistralai"],
    check=True)

import torch
import wandb
import anthropic
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
# ===== INLINE REWARD (standalone — no cross-module imports) =====
import re

def _score_syntax(code):
    """Simplified syntax scoring."""
    issues = 0
    # Strip comments/strings for structural checks
    stripped = re.sub(r"--\[\[.*?\]\]", "", code, flags=re.DOTALL)
    stripped = re.sub(r"--[^\n]*", "", stripped)
    stripped = re.sub(r'"(?:[^"\\]|\\.)*"', '""', stripped)
    stripped = re.sub(r"'(?:[^'\\]|\\.)*'", "''", stripped)

    # Bracket check
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for ch in stripped:
        if ch in pairs:
            stack.append(ch)
        elif ch in pairs.values():
            if not stack:
                issues += 1; break
            opener = stack.pop()
            if pairs[opener] != ch:
                issues += 1; break
    if len(stack) > 2:
        issues += 1

    # Block balance
    funcs = len(re.findall(r"\bfunction\b", stripped))
    ifs = len(re.findall(r"\bif\b", stripped)) - len(re.findall(r"\belseif\b", stripped))
    fors = len(re.findall(r"\bfor\b", stripped))
    whiles = len(re.findall(r"\bwhile\b", stripped))
    ends = len(re.findall(r"\bend\b", stripped))
    openers = funcs + ifs + fors + whiles
    if abs(ends - openers) > 2:
        issues += 1

    # Python-isms
    for p in [r"\bdef\s+\w+\s*\(", r"\bclass\s+\w+", r"^import\s+\w+",
              r"\bTrue\b", r"\bFalse\b", r"\bNone\b", r"\belif\b"]:
        if re.search(p, stripped, re.MULTILINE):
            issues += 1

    return max(0.0, 1.0 - issues * 0.12)


def _score_api(code):
    """Simplified API scoring."""
    issues = 0
    deprecated = [
        (r"\bwait\s*\(", r"\btask\.wait\s*\("),
        (r"\bspawn\s*\(", r"\btask\.spawn\s*\("),
        (r"game\.Players\b", r'GetService\s*\(\s*"Players"\s*\)'),
        (r"game\.Workspace\b", r'GetService\s*\(\s*"Workspace"\s*\)'),
    ]
    for dep, modern in deprecated:
        if re.search(dep, code) and not re.search(modern, code):
            issues += 1
    good = sum(1 for p in [r"\.Connect\s*\(", r"Instance\.new", r"\btask\.\w+",
                            r"GetService"] if re.search(p, code))
    return max(0.0, min(1.0, 1.0 - issues * 0.15 + good * 0.05))


def _score_bugs(code):
    """Simplified bug scoring."""
    stripped = re.sub(r"--[^\n]*", "", code)
    penalty = 0.0
    # Unchecked FindFirstChild
    penalty += len(re.findall(
        r":FindFirstChild\s*\([^)]+\)\s*[\.\:]", stripped)) * 0.20
    # Infinite loop
    if re.search(r"\bwhile\s+true\s+do\b", stripped):
        block = stripped[stripped.find("while true do"):]
        if not re.search(r"\btask\.wait\b|\bwait\b|\bRunService\b|\b:Wait\b", block[:500]):
            penalty += 0.30
    # DataStore without pcall
    ds_calls = len(re.findall(r"(?:GetAsync|SetAsync|UpdateAsync)\s*\(", stripped))
    if ds_calls and not re.search(r"\bpcall\b|\bxpcall\b", stripped):
        penalty += 0.20
    return max(0.0, 1.0 - penalty)


def _score_quality(code):
    """Simplified quality scoring."""
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    total = max(len(non_empty), 1)
    comments = sum(1 for l in lines if re.match(r"\s*--", l))
    density = comments / total
    comment_s = 1.0 if 0.05 <= density <= 0.35 else (0.5 if comments > 0 else 0.2)
    has_pcall = bool(re.search(r"\bpcall\b", code))
    needs_pcall = bool(re.search(r"DataStore|HttpService|GetAsync|SetAsync", code))
    err_s = 1.0 if has_pcall else (0.15 if needs_pcall else 0.75)
    len_s = 1.0 if 15 <= total <= 300 else (0.6 if total >= 8 else 0.3)
    return comment_s * 0.25 + err_s * 0.25 + len_s * 0.25 + 0.25  # naming baseline


def compute_reward(code):
    """Compute 0-1 reward from 4 deterministic scorers."""
    try:
        return round(
            _score_syntax(code) * 0.25 +
            _score_api(code) * 0.25 +
            _score_bugs(code) * 0.25 +
            _score_quality(code) * 0.25, 4)
    except Exception:
        return 0.0

# ===== CONFIG =====
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
WANDB_KEY = (
    "wandb_v1_LWsj8RsQTjAFkGgvtIQV8RP0NNL"
    "_ZuBdEJIWiOCUvWXccxtXlYlJh3jNMvqIDSy"
    "fAPwfNM51tyDHg")
HF_SFT_REPO = "squaredcuber/roblox-luau-mistral-7b-2"
HF_RFT_REPO = "squaredcuber/roblox-luau-mistral-7b-rft"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TRAINING_DATA = PROJECT_ROOT / "data" / "training_data.jsonl"
NUM_PROMPTS = 200
NUM_GENERATIONS = 4
LLM_THRESHOLD = 0.7
DET_THRESHOLD = 0.85
# ==================

os.environ["WANDB_API_KEY"] = WANDB_KEY
os.environ["WANDB_PROJECT"] = "roblox-luau-codegen"
os.environ["WANDB_ENTITY"] = "carpediemhari-n-a"


def judge_code(client, task, code):
    """Score code 0-1 using Claude Sonnet."""
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=20,
            messages=[{"role": "user", "content": (
                "Rate this Roblox Luau code 0-10.\n"
                "0=gibberish/wrong language/not code\n"
                "3=valid Lua but not Roblox Luau\n"
                "5=valid Luau but incomplete/buggy\n"
                "7=functional but missing best practices\n"
                "10=production-ready Roblox script\n\n"
                f"Task: {task}\n\n"
                f"Code:\n{code[:2500]}\n\n"
                "Reply with ONLY a number 0-10.")}])
        return float(resp.content[0].text.strip()) / 10
    except Exception as e:
        print(f"    Judge error: {e}")
        return 0.0


def main():
    if not ANTHROPIC_KEY:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    # Load training data
    print("=== Loading training data ===")
    with open(TRAINING_DATA, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(rows)} samples")

    # Init wandb
    wandb.init(
        project="roblox-luau-codegen",
        entity="carpediemhari-n-a",
        name="rft-mistral7b-sonnet-judge",
        tags=["rft", "self-improve", "sonnet-judge"])

    # Load SFT model from HuggingFace
    print(f"\n=== Loading SFT model from {HF_SFT_REPO} ===")
    print(f"Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    model_gen = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map="auto")
    model_gen = PeftModel.from_pretrained(
        model_gen, HF_SFT_REPO)
    model_gen = model_gen.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token

    # Quick sanity check
    print("\n=== Sanity check ===")
    test_inp = tok.apply_chat_template(
        [{"role": "system",
          "content": "You are an expert Roblox Luau programmer. Output ONLY Luau code."},
         {"role": "user",
          "content": "Create a simple kill brick"}],
        tokenize=False, add_generation_prompt=True)
    test_toks = tok(test_inp, return_tensors="pt").to("cuda")
    test_out = model_gen.generate(
        **test_toks, max_new_tokens=200,
        do_sample=True, temperature=0.3)
    test_code = tok.decode(test_out[0], skip_special_tokens=True)
    if "[/INST]" in test_code:
        test_code = test_code.split("[/INST]")[-1].strip()
    print(test_code[:300])
    print("---")

    # Step 1: Generate + filter with Sonnet judge
    print(f"\n=== Generating candidates ({NUM_PROMPTS} prompts x {NUM_GENERATIONS} each) ===")
    candidates = []
    total_judged = 0
    total_generated = 0

    for i, r in enumerate(rows[:NUM_PROMPTS]):
        prompt = None
        sys_msg = None
        for msg in r["messages"]:
            if msg["role"] == "user":
                prompt = msg["content"]
            if msg["role"] == "system":
                sys_msg = msg["content"]
        if not prompt:
            continue

        inp = tok.apply_chat_template(
            [{"role": "system",
              "content": sys_msg or (
                  "You are an expert Roblox Luau "
                  "programmer. Output ONLY complete "
                  "Luau code. No explanations.")},
             {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True)
        toks_in = tok(inp, return_tensors="pt").to("cuda")

        outs = model_gen.generate(
            **toks_in, max_new_tokens=512,
            do_sample=True, temperature=0.9,
            num_return_sequences=NUM_GENERATIONS)

        for out in outs:
            code = tok.decode(out, skip_special_tokens=True)
            if "[/INST]" in code:
                code = code.split("[/INST]")[-1].strip()
            total_generated += 1

            # Quick pre-filter (free)
            if len(code.strip()) < 150:
                continue
            if "function" not in code:
                continue

            # Sonnet judge
            llm_score = judge_code(client, prompt, code)
            det_score = compute_reward(code)
            total_judged += 1

            # Must pass BOTH
            if llm_score >= LLM_THRESHOLD and det_score >= DET_THRESHOLD:
                candidates.append({
                    "text": (
                        f"<s>[INST] "
                        f"{sys_msg or 'You are an expert Roblox Luau programmer.'}"
                        f"\n\n{prompt} [/INST]\n"
                        f"{code}</s>"),
                    "score": det_score,
                    "llm_score": llm_score})

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{NUM_PROMPTS} — generated {total_generated}, "
                  f"judged {total_judged}, kept {len(candidates)}")

    print(f"\nGenerated {total_generated}, judged {total_judged}, "
          f"kept {len(candidates)}")

    if len(candidates) < 10:
        print("WARNING: Too few candidates passed. Lowering thresholds.")
        # Don't exit — still useful data

    # Save candidates for inspection
    candidates_path = PROJECT_ROOT / "results" / "rft_candidates.json"
    candidates_path.parent.mkdir(exist_ok=True)
    with open(candidates_path, "w") as f:
        json.dump([{"score": c["score"], "llm_score": c["llm_score"],
                     "text": c["text"][:200]} for c in candidates],
                  f, indent=2)
    print(f"Candidates saved to {candidates_path}")

    # Step 2: Free GPU
    print("\n=== Freeing GPU for training ===")
    del model_gen
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    # Step 3: Train on best outputs
    print(f"\n=== Training on {len(candidates)} samples ===")
    rft_ds = Dataset.from_list(candidates)

    model_rft = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map="auto")

    rft_cfg = SFTConfig(
        output_dir="/tmp/rft-mistral7b",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True, logging_steps=5,
        report_to="wandb",
        run_name="rft-mistral7b-sonnet-judge",
        dataset_text_field="text")

    rft_lora = LoraConfig(
        r=32, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM")

    trainer_rft = SFTTrainer(
        model=model_rft, args=rft_cfg,
        train_dataset=rft_ds,
        processing_class=tok,
        peft_config=rft_lora)

    trainer_rft.train()
    trainer_rft.save_model("/tmp/rft-mistral7b/final")
    tok.save_pretrained("/tmp/rft-mistral7b/final")

    # Step 4: Test output
    print("\n=== Testing RFT model ===")
    test_inp = tok.apply_chat_template(
        [{"role": "system",
          "content": "You are an expert Roblox Luau programmer. Output ONLY Luau code."},
         {"role": "user",
          "content": "Create a coin collection system"}],
        tokenize=False, add_generation_prompt=True)
    test_toks = tok(test_inp, return_tensors="pt").to("cuda")
    test_out = model_rft.generate(
        **test_toks, max_new_tokens=300,
        do_sample=True, temperature=0.3)
    print(tok.decode(test_out[0], skip_special_tokens=True))

    # Step 5: Push to HuggingFace
    print(f"\n=== Pushing to {HF_RFT_REPO} ===")
    try:
        from huggingface_hub import HfApi
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            api = HfApi(token=hf_token)
            api.create_repo(HF_RFT_REPO,
                exist_ok=True, repo_type="model",
                token=hf_token)
            api.upload_folder(
                folder_path="/tmp/rft-mistral7b/final",
                repo_id=HF_RFT_REPO, token=hf_token)
            print("Pushed to HuggingFace!")
        else:
            print("No HF_TOKEN set, skipping push")
    except Exception as e:
        print(f"HF push failed: {e}")

    wandb.finish()
    print("\n=== RFT PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
