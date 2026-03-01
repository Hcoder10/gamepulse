"""Analyze evaluation results to identify weaknesses below thresholds."""

THRESHOLDS = {
    "syntax": 0.85,
    "api": 0.80,
    "bugs": 0.80,
    "quality": 0.75,
    "task": 0.70,
}

WEIGHTS = {
    "syntax": 0.20,
    "api": 0.20,
    "bugs": 0.20,
    "quality": 0.15,
    "task": 0.25,
}

# Maps Weave result keys to our dimension names
SCORE_KEY_MAP = {
    "SyntaxScorer": ("syntax", "syntax_score"),
    "ApiScorer": ("api", "api_score"),
    "BugScorer": ("bugs", "bug_score"),
    "QualityScorer": ("quality", "quality_score"),
    "TaskCompletionScorer": ("task", "task_completion_score"),
}


def extract_dimension_scores(results: dict) -> dict[str, float]:
    """Extract per-dimension average scores from Weave evaluation results."""
    scores = {}

    for scorer_name, (dim_name, score_key) in SCORE_KEY_MAP.items():
        scorer_data = results.get(scorer_name, {})
        # Weave returns mean values in the scorer dict
        if isinstance(scorer_data, dict):
            value = scorer_data.get(score_key, {})
            if isinstance(value, dict):
                scores[dim_name] = value.get("mean", 0.5)
            elif isinstance(value, (int, float)):
                scores[dim_name] = float(value)
            else:
                scores[dim_name] = 0.5
        else:
            scores[dim_name] = 0.5

    return scores


def compute_composite(scores: dict[str, float]) -> float:
    """Compute weighted composite score."""
    total = sum(scores.get(dim, 0.0) * w for dim, w in WEIGHTS.items())
    return round(total, 4)


def identify_weaknesses(scores: dict[str, float]) -> list[dict]:
    """Find dimensions below their threshold, ranked by gap (worst first)."""
    weaknesses = []
    for dim, threshold in THRESHOLDS.items():
        actual = scores.get(dim, 0.0)
        if actual < threshold:
            gap = threshold - actual
            weaknesses.append({
                "dimension": dim,
                "score": round(actual, 4),
                "threshold": threshold,
                "gap": round(gap, 4),
            })

    weaknesses.sort(key=lambda w: w["gap"], reverse=True)
    return weaknesses


def analyze_results(results: dict) -> dict:
    """Full analysis: scores, composite, weaknesses."""
    scores = extract_dimension_scores(results)
    composite = compute_composite(scores)
    weaknesses = identify_weaknesses(scores)

    return {
        "dimension_scores": scores,
        "composite_score": composite,
        "weaknesses": weaknesses,
        "all_passing": len(weaknesses) == 0,
    }
