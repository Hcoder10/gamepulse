"""Self-Correction Pipeline: The Code Doctor.

The model generates code, our scorers diagnose issues,
then the model rewrites its own code to fix those specific issues.
The corrected versions become new training data.
"""

import re
from src.mistral_client import generate_completion
from scorers.syntax_scorer import SyntaxScorer
from scorers.api_scorer import ApiScorer
from scorers.bug_scorer import BugScorer
from scorers.quality_scorer import QualityScorer


_syntax = SyntaxScorer()
_api = ApiScorer()
_bug = BugScorer()
_quality = QualityScorer()


def diagnose(code: str) -> dict:
    """Run all scorers and return a structured diagnosis."""
    s_syntax = _syntax.score(code)
    s_api = _api.score(code)
    s_bug = _bug.score(code)
    s_quality = _quality.score(code)

    issues = []
    issues.extend(s_syntax.get("syntax_issues", []))
    issues.extend(s_api.get("api_issues", []))
    for bug in s_bug.get("bugs_found", []):
        issues.append(bug["bug"])

    scores = {
        "syntax": s_syntax.get("syntax_score", 0),
        "api": s_api.get("api_score", 0),
        "bugs": s_bug.get("bug_score", 0),
        "quality": s_quality.get("quality_score", 0),
    }
    avg = sum(scores.values()) / len(scores)

    return {
        "scores": scores,
        "avg_score": round(avg, 3),
        "issues": issues,
        "passed": avg >= 0.85,
    }


def format_diagnosis_report(diagnosis: dict) -> str:
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


def heal(task_description: str, broken_code: str, diagnosis: dict) -> str:
    """Ask the model to fix its own code based on the diagnosis.

    This is the self-correction step: the model sees its own errors
    and rewrites the code to fix them.
    """
    error_report = format_diagnosis_report(diagnosis)

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

    healed = generate_completion(system, heal_prompt, temperature=0.3)

    # Strip markdown fences if present
    healed = re.sub(r"^```(?:lua|luau)?\s*\n?", "", healed.strip())
    healed = re.sub(r"\n?```\s*$", "", healed)

    return healed.strip()


def self_correct(task_description: str, code: str, max_rounds: int = 3) -> dict:
    """Run the full self-correction loop.

    Generate -> Diagnose -> Heal -> Re-diagnose -> until passing or max rounds.

    Returns dict with correction history and final code.
    """
    history = []
    current_code = code

    for round_num in range(1, max_rounds + 1):
        diag = diagnose(current_code)
        history.append({
            "round": round_num,
            "code": current_code,
            "diagnosis": diag,
        })

        if diag["passed"]:
            break

        if round_num < max_rounds:
            current_code = heal(task_description, current_code, diag)

    final_diag = diagnose(current_code)

    return {
        "original_code": code,
        "final_code": current_code,
        "rounds": len(history),
        "original_score": history[0]["diagnosis"]["avg_score"],
        "final_score": final_diag["avg_score"],
        "improved": final_diag["avg_score"] > history[0]["diagnosis"]["avg_score"],
        "passed": final_diag["passed"],
        "history": history,
    }
