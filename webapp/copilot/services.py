"""
Service layer for code generation, scoring, and self-correction.

Supports inference backends (checked in order):
1. MODEL_ENDPOINT_URL — vLLM/TGI server serving LoRA adapters
2. SFT_ENDPOINT_URL / RFT_ENDPOINT_URL — dedicated HF Inference Endpoints
3. Mistral API — fallback using fine-tuned prompt versions
"""
import re
import json
import time
import logging
from pathlib import Path

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import scorers from the parent project (added to sys.path in settings.py)
# ---------------------------------------------------------------------------
from scorers.syntax_scorer import SyntaxScorer
from scorers.api_scorer import ApiScorer
from scorers.bug_scorer import BugScorer
from scorers.quality_scorer import QualityScorer

_syntax_scorer = SyntaxScorer()
_api_scorer = ApiScorer()
_bug_scorer = BugScorer()
_quality_scorer = QualityScorer()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert Roblox Luau programmer. Generate complete, production-ready Luau scripts.

Structure every script like this:
-- Brief description of what this script does
local Services = game:GetService("ServiceName")
-- Configuration
-- Functions
-- Main logic

Rules:

1. Services: Access via game:GetService("Players"), never game.Players directly.

2. Variables: Declare everything with `local`. No globals.

3. Modern API only:
   - task.wait() / task.spawn() / task.delay() instead of wait() / spawn() / delay()
   - .Connect() with capital C, never .connect()
   - Instance.new("Part") then set .Parent separately, never Instance.new("Part", parent)

4. Safety patterns:
   - Wrap DataStore calls in pcall: local ok, result = pcall(function() return store:GetAsync(key) end)
   - Check FindFirstChild before accessing: local obj = parent:FindFirstChild("Name") if obj then ... end
   - Every while true do loop must contain task.wait() or another yield
   - Use WaitForChild() for objects that load asynchronously

5. Naming: PascalCase for services (local Players = ...), camelCase for variables and functions.

6. Comments: Start with a one-line description comment. Add a brief comment before each major section.

7. Completeness: Implement every feature in the task. No placeholders, no TODOs, no "add your code here". Every function must have a full body.

8. Cleanup: Store connections in variables. Disconnect when appropriate. Destroy unused instances.

Output only the Luau code. No markdown fences, no explanations.\
"""


def _strip_markdown_fences(code: str) -> str:
    """Remove markdown code fences if present."""
    code = code.strip()
    code = re.sub(r"^```(?:lua|luau)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


# ---------------------------------------------------------------------------
# Backend 1: vLLM / TGI server (OpenAI-compatible)
# Set MODEL_ENDPOINT_URL to use this. Serves LoRA adapters.
# ---------------------------------------------------------------------------

def _call_model_endpoint(task_description: str, model_name: str) -> str:
    """Call a vLLM/TGI OpenAI-compatible endpoint with LoRA adapters."""
    endpoint_url = getattr(settings, "MODEL_ENDPOINT_URL", "")
    if not endpoint_url:
        raise ValueError("MODEL_ENDPOINT_URL is not configured.")

    url = f"{endpoint_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_description},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return _strip_markdown_fences(content)
        except Exception as e:
            if attempt == 2:
                raise
            logger.warning(f"Model endpoint retry {attempt + 1}: {e}")
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Backend 2: Dedicated HF Inference Endpoints
# Set SFT_ENDPOINT_URL / RFT_ENDPOINT_URL to use these.
# ---------------------------------------------------------------------------

def _call_hf_endpoint(task_description: str, endpoint_url: str) -> str:
    """Call an HF Inference Endpoint (OpenAI-compatible or text-generation)."""
    hf_token = getattr(settings, "HF_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    # Try OpenAI-compatible chat format first
    chat_url = f"{endpoint_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_description},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
        "stream": False,
    }

    for attempt in range(3):
        try:
            resp = requests.post(chat_url, json=payload, headers=headers, timeout=180)
            if resp.status_code == 404:
                # Endpoint doesn't support /v1/chat/completions, try raw text-generation
                return _call_hf_endpoint_text(task_description, endpoint_url, headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return _strip_markdown_fences(content)
        except requests.exceptions.HTTPError:
            if resp.status_code == 404:
                return _call_hf_endpoint_text(task_description, endpoint_url, headers)
            if attempt == 2:
                raise
            logger.warning(f"HF endpoint retry {attempt + 1}")
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == 2:
                raise
            logger.warning(f"HF endpoint retry {attempt + 1}: {e}")
            time.sleep(2 ** attempt)


def _call_hf_endpoint_text(task_description: str, endpoint_url: str, headers: dict) -> str:
    """Fallback: call HF endpoint with text-generation format ([INST] prompt)."""
    prompt = f"[INST] {SYSTEM_PROMPT}\n\n{task_description} [/INST]"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 4096,
            "temperature": 0.7,
            "return_full_text": False,
        },
    }

    resp = requests.post(endpoint_url, json=payload, headers=headers, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and len(data) > 0:
        content = data[0].get("generated_text", "")
    else:
        content = str(data)
    return _strip_markdown_fences(content)


# ---------------------------------------------------------------------------
# Backend 3: Mistral API with fine-tuned prompts (fallback)
# ---------------------------------------------------------------------------

MISTRAL_MODEL = "mistral-small-latest"

# Fine-tuned prompt versions (loaded lazily)
_FALLBACK_PROMPTS = {
    "sft": None,   # loaded lazily from v1.txt
    "rft": None,   # loaded lazily from v7.txt (best evolved prompt)
}


def _load_prompt_version(version: str) -> str:
    """Load a prompt version from configs/prompt_versions/."""
    path = Path(settings.BASE_DIR).parent / "configs" / "prompt_versions" / f"{version}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return SYSTEM_PROMPT


def _get_fallback_prompt(model_choice: str) -> str:
    """Get the fine-tuned prompt for Mistral API fallback mode."""
    if model_choice == "sft":
        if _FALLBACK_PROMPTS["sft"] is None:
            _FALLBACK_PROMPTS["sft"] = _load_prompt_version("v1")
        return _FALLBACK_PROMPTS["sft"]
    # Default to RFT (best evolved prompt)
    if _FALLBACK_PROMPTS["rft"] is None:
        _FALLBACK_PROMPTS["rft"] = _load_prompt_version("v7")
    return _FALLBACK_PROMPTS["rft"]


def _call_mistral(task_description: str, system_prompt: str, model: str = None) -> str:
    """Call Mistral API with a given system prompt and model."""
    api_key = settings.MISTRAL_API_KEY
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not configured.")

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_description},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return _strip_markdown_fences(content)
        except Exception as e:
            if attempt == 2:
                raise
            logger.warning(f"Mistral API retry {attempt + 1}: {e}")
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Code generation dispatcher
# ---------------------------------------------------------------------------

def generate_code(task_description: str, model_choice: str = "rft") -> str:
    """
    Generate Luau code using fine-tuned models. Tries backends in order:

    1. MODEL_ENDPOINT_URL (vLLM/TGI with LoRA adapters)
    2. SFT_ENDPOINT_URL / RFT_ENDPOINT_URL (HF Inference Endpoints)
    3. Mistral API with fine-tuned prompt versions
    """
    if model_choice not in ("sft", "rft"):
        model_choice = "rft"

    # Backend 1: vLLM/TGI serving LoRA adapters
    endpoint_url = getattr(settings, "MODEL_ENDPOINT_URL", "")
    if endpoint_url:
        return _call_model_endpoint(task_description, model_choice)

    # Backend 2: Dedicated HF Inference Endpoints per model
    if model_choice == "sft":
        sft_url = getattr(settings, "SFT_ENDPOINT_URL", "")
        if sft_url:
            return _call_hf_endpoint(task_description, sft_url)
    elif model_choice == "rft":
        rft_url = getattr(settings, "RFT_ENDPOINT_URL", "")
        if rft_url:
            return _call_hf_endpoint(task_description, rft_url)

    # Backend 3: Mistral API with fine-tuned prompt versions
    logger.info(f"No model endpoint configured for '{model_choice}', using Mistral API with evolved prompts")
    return _call_mistral(task_description, _get_fallback_prompt(model_choice))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_code(code: str) -> dict:
    """
    Run all four scorers on a piece of code.
    Returns a dict with individual and average scores plus issues.
    """
    s_syntax = _syntax_scorer.score(output=code)
    s_api = _api_scorer.score(output=code)
    s_bug = _bug_scorer.score(output=code)
    s_quality = _quality_scorer.score(output=code)

    scores = {
        "syntax": s_syntax.get("syntax_score", 0),
        "api": s_api.get("api_score", 0),
        "bugs": s_bug.get("bug_score", 0),
        "quality": s_quality.get("quality_score", 0),
    }
    avg = sum(scores.values()) / len(scores)

    issues = []
    issues.extend(s_syntax.get("syntax_issues", []))
    issues.extend(s_api.get("api_issues", []))
    for bug in s_bug.get("bugs_found", []):
        issues.append(bug["bug"])

    return {
        "scores": scores,
        "avg_score": round(avg, 3),
        "issues": issues,
        "passed": avg >= 0.85,
    }


# ---------------------------------------------------------------------------
# Self-correction (diagnosis + healing via Mistral)
# ---------------------------------------------------------------------------

def _format_diagnosis_report(diagnosis: dict) -> str:
    """Format diagnosis into a readable error report for the model."""
    lines = ["ERRORS FOUND IN YOUR CODE:"]
    for i, issue in enumerate(diagnosis["issues"], 1):
        lines.append(f"{i}. {issue}")
    lines.append("")
    lines.append("SCORES:")
    for dim, score in diagnosis["scores"].items():
        status = "PASS" if score >= 0.85 else "FAIL"
        lines.append(f"  {dim}: {score:.2f} [{status}]")
    return "\n".join(lines)


def heal_code(task_description: str, broken_code: str, diagnosis: dict) -> str:
    """Ask Mistral to fix code based on the diagnosis."""
    api_key = settings.MISTRAL_API_KEY
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not configured.")

    error_report = _format_diagnosis_report(diagnosis)

    heal_prompt = f"""You wrote this Roblox Luau script but it has issues. Fix ALL the errors listed below.

ORIGINAL TASK: {task_description}

YOUR BROKEN CODE:
```lua
{broken_code}
```

{error_report}

RULES FOR THE FIX:
- Fix every listed error
- Wrap ALL DataStore calls (GetAsync/SetAsync/UpdateAsync) in pcall
- Check FindFirstChild results before accessing properties
- Every while true do loop must contain task.wait() or a yield
- Use game:GetService(), not game.ServiceName
- Use task.wait() not wait(), task.spawn() not spawn()
- Keep all variables local
- Output ONLY the corrected Luau code, no markdown fences"""

    system = (
        "You are a Roblox Luau code reviewer. "
        "Fix the code to resolve all listed issues. "
        "Output only corrected Luau code."
    )

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": heal_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _strip_markdown_fences(content)


def self_correct(task_description: str, code: str, max_rounds: int = 3) -> dict:
    """
    Run the full self-correction loop.
    Generate -> Diagnose -> Heal -> Re-diagnose until passing or max_rounds.
    """
    history = []
    current_code = code

    for round_num in range(1, max_rounds + 1):
        diag = score_code(current_code)
        history.append({
            "round": round_num,
            "code": current_code,
            "scores": diag["scores"],
            "avg_score": diag["avg_score"],
            "issues": diag["issues"],
            "passed": diag["passed"],
        })

        if diag["passed"]:
            break

        if round_num < max_rounds:
            current_code = heal_code(task_description, current_code, diag)

    final_diag = score_code(current_code)

    return {
        "original_code": code,
        "final_code": current_code,
        "rounds": len(history),
        "original_scores": history[0]["scores"],
        "original_avg": history[0]["avg_score"],
        "final_scores": final_diag["scores"],
        "final_avg": final_diag["avg_score"],
        "improved": final_diag["avg_score"] > history[0]["avg_score"],
        "passed": final_diag["passed"],
        "history": history,
    }


# ---------------------------------------------------------------------------
# Results data loaders
# ---------------------------------------------------------------------------

def load_iteration_log() -> list:
    """Load results/iteration_log.json."""
    path = settings.RESULTS_DIR / "iteration_log.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def load_forge_log() -> list:
    """Load results/forge_log.json."""
    path = settings.RESULTS_DIR / "forge_log.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def get_test_tasks() -> list:
    """Return the preset test tasks for the compare page."""
    from data.test_tasks import TEST_TASKS
    return TEST_TASKS


# ---------------------------------------------------------------------------
# Pipeline data loaders
# ---------------------------------------------------------------------------

def load_prompt_versions() -> list:
    """Load all prompt versions from configs/prompt_versions/."""
    versions_dir = Path(settings.BASE_DIR).parent / "configs" / "prompt_versions"
    versions = []
    if versions_dir.exists():
        for f in sorted(versions_dir.glob("v*.txt"), key=lambda p: _version_sort_key(p)):
            content = f.read_text(encoding="utf-8").strip()
            versions.append({
                "version": f.stem,
                "filename": f.name,
                "content": content,
                "length": len(content),
            })
    return versions


def _version_sort_key(path: Path) -> int:
    """Extract version number for sorting (v1.txt -> 1)."""
    try:
        return int(path.stem.lstrip("v"))
    except ValueError:
        return 999


def get_prompt_version(version: str) -> str | None:
    """Load a specific prompt version's content."""
    path = Path(settings.BASE_DIR).parent / "configs" / "prompt_versions" / f"{version}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def get_pipeline_status() -> dict:
    """Aggregate pipeline status for the dashboard."""
    iteration_log = load_iteration_log()
    forge_log = load_forge_log()
    prompt_versions = load_prompt_versions()

    best_score = 0.0
    best_version = "v1"
    latest_scores = {}

    if iteration_log:
        for entry in iteration_log:
            if entry.get("composite_score", 0) > best_score:
                best_score = entry["composite_score"]
                best_version = entry.get("prompt_version", "v1")
        latest = iteration_log[-1]
        latest_scores = latest.get("dimension_scores", {})

    return {
        "iterations_run": len(iteration_log),
        "best_composite_score": round(best_score, 4),
        "best_version": best_version,
        "latest_version": prompt_versions[-1]["version"] if prompt_versions else "v1",
        "total_versions": len(prompt_versions),
        "latest_scores": latest_scores,
        "forge_rounds": len(forge_log),
        "latest_forge_pass_rate": forge_log[-1].get("pass_rate", 0) if forge_log else 0,
    }


# ---------------------------------------------------------------------------
# Studio Plugin Command Queue (in-memory)
# ---------------------------------------------------------------------------
import threading
import uuid
from collections import defaultdict

_studio_lock = threading.Lock()
_studio_commands = defaultdict(list)  # session_id -> [commands]
_studio_results = {}  # command_id -> result
_studio_last_poll = {}  # session_id -> timestamp


def queue_studio_command(session_id: str, command_type: str, payload: dict) -> str:
    """Queue a command for the Studio plugin to pick up."""
    cmd_id = str(uuid.uuid4())[:8]
    cmd = {
        "id": cmd_id,
        "type": command_type,
        "payload": payload,
        "status": "pending",
    }
    with _studio_lock:
        _studio_commands[session_id].append(cmd)
    return cmd_id


def poll_studio_commands(session_id: str) -> list:
    """Return and clear pending commands for a session."""
    with _studio_lock:
        cmds = _studio_commands.pop(session_id, [])
        _studio_last_poll[session_id] = time.time()
    return cmds


def get_studio_last_poll(session_id: str) -> float:
    """Get the timestamp of the last poll for a session."""
    with _studio_lock:
        return _studio_last_poll.get(session_id, 0)


def report_studio_result(command_id: str, success: bool, message: str):
    """Store result from Studio plugin."""
    with _studio_lock:
        _studio_results[command_id] = {
            "success": success,
            "message": message,
        }


def get_studio_result(command_id: str) -> dict:
    """Get result for a command."""
    with _studio_lock:
        return _studio_results.get(command_id)


def agentic_generate(task_description: str, model_choice: str = "rft",
                     auto_correct: bool = True, session_id: str = None) -> dict:
    """Full agentic pipeline: generate -> score -> self-correct -> queue for Studio.

    Returns a stream-like list of steps for the chat UI.
    """
    steps = []

    # Step 1: Generate
    steps.append({"type": "thinking", "text": f"Generating code with {model_choice.upper()} model..."})
    code = generate_code(task_description, model_choice)
    steps.append({"type": "code", "code": code, "label": "Generated Code"})

    # Step 2: Score
    steps.append({"type": "thinking", "text": "Running quality scorers..."})
    diag = score_code(code)
    steps.append({"type": "scores", "scores": diag["scores"], "avg_score": diag["avg_score"],
                   "issues": diag["issues"], "passed": diag["passed"]})

    # Step 3: Self-correct if needed
    if auto_correct and not diag["passed"]:
        steps.append({"type": "thinking", "text": f"Score {diag['avg_score']:.0%} below threshold. Self-correcting..."})
        result = self_correct(task_description, code, max_rounds=3)
        code = result["final_code"]
        steps.append({"type": "correction", "original_scores": result["original_scores"],
                       "final_scores": result["final_scores"], "original_avg": result["original_avg"],
                       "final_avg": result["final_avg"], "rounds": result["rounds"],
                       "improved": result["improved"]})
        steps.append({"type": "code", "code": code, "label": "Corrected Code"})
        final_diag = score_code(code)
        steps.append({"type": "scores", "scores": final_diag["scores"], "avg_score": final_diag["avg_score"],
                       "issues": final_diag["issues"], "passed": final_diag["passed"]})
    else:
        if diag["passed"]:
            steps.append({"type": "thinking", "text": "All quality checks passed!"})

    # Step 4: Queue for Studio if connected
    if session_id:
        cmd_id = queue_studio_command(session_id, "insert_script", {
            "name": _derive_script_name(task_description),
            "source": code,
            "parent": "ServerScriptService",
            "class": "Script",
        })
        steps.append({"type": "studio", "command_id": cmd_id,
                       "text": "Script queued for Roblox Studio"})

    steps.append({"type": "done", "final_code": code})
    return {"steps": steps, "final_code": code}


def _derive_script_name(task: str) -> str:
    """Generate a PascalCase script name from a task description."""
    words = task.split()[:4]
    name = "".join(w.capitalize() for w in words if w.isalpha())
    return name or "GeneratedScript"
