"""Reward function for GRPO using our 5 scorers.

This is the key differentiator — the same scorers that evaluate
also provide the reward signal for RL fine-tuning.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scorers.syntax_scorer import SyntaxScorer
from scorers.api_scorer import ApiScorer
from scorers.bug_scorer import BugScorer
from scorers.quality_scorer import QualityScorer


# Scorer weights for reward signal
REWARD_WEIGHTS = {
    "syntax": 0.25,
    "api": 0.25,
    "bugs": 0.25,
    "quality": 0.25,
}

# Initialize scorers once
_syntax_scorer = SyntaxScorer()
_api_scorer = ApiScorer()
_bug_scorer = BugScorer()
_quality_scorer = QualityScorer()


def compute_reward(code: str) -> float:
    """Compute a scalar reward from our scorers.

    Returns a float in [0, 1] representing code quality.
    Used as the reward signal in GRPO training.
    """
    try:
        s_syntax = _syntax_scorer.score(code)
        s_api = _api_scorer.score(code)
        s_bug = _bug_scorer.score(code)
        s_quality = _quality_scorer.score(code)

        reward = (
            REWARD_WEIGHTS["syntax"] * s_syntax.get("syntax_score", 0)
            + REWARD_WEIGHTS["api"] * s_api.get("api_score", 0)
            + REWARD_WEIGHTS["bugs"] * s_bug.get("bug_score", 0)
            + REWARD_WEIGHTS["quality"] * s_quality.get("quality_score", 0)
        )

        return round(reward, 4)
    except Exception:
        return 0.0


def compute_reward_batch(codes: list[str]) -> list[float]:
    """Compute rewards for a batch of code completions."""
    return [compute_reward(code) for code in codes]


def compute_reward_detailed(code: str) -> dict:
    """Compute reward with per-dimension breakdown."""
    try:
        s_syntax = _syntax_scorer.score(code)
        s_api = _api_scorer.score(code)
        s_bug = _bug_scorer.score(code)
        s_quality = _quality_scorer.score(code)

        scores = {
            "syntax": s_syntax.get("syntax_score", 0),
            "api": s_api.get("api_score", 0),
            "bugs": s_bug.get("bug_score", 0),
            "quality": s_quality.get("quality_score", 0),
        }

        reward = sum(REWARD_WEIGHTS[k] * v for k, v in scores.items())

        return {
            "reward": round(reward, 4),
            "scores": scores,
            "issues": {
                "syntax": s_syntax.get("syntax_issues", []),
                "api": s_api.get("api_issues", []),
                "bugs": [b["bug"] for b in s_bug.get("bugs_found", [])],
            },
        }
    except Exception as e:
        return {"reward": 0.0, "scores": {}, "issues": {"error": str(e)}}
