import asyncio
import weave
from data.test_tasks import TEST_TASKS
from scorers import SyntaxScorer, ApiScorer, BugScorer, QualityScorer, TaskCompletionScorer


def build_dataset() -> weave.Dataset:
    """Build and publish a Weave Dataset from test tasks."""
    rows = [
        {"task_description": t["task_description"], "category": t["category"]}
        for t in TEST_TASKS
    ]
    dataset = weave.Dataset(name="roblox-luau-tasks", rows=rows)
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


async def run_evaluation(generator, iteration: int = 1, version: int = 1) -> dict:
    """Run evaluation and return aggregated results."""
    dataset = build_dataset()
    scorers = build_scorers()

    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=scorers,
        name=f"luau-eval-iter{iteration}-v{version}",
    )

    results = await evaluation.evaluate(generator)
    return results


async def run_evaluation_with_details(generator, iteration: int = 1, version: int = 1) -> tuple[dict, list[dict]]:
    """Run evaluation and also collect per-task outputs + scores for example-driven improvement.

    Returns (aggregated_results, per_task_details).
    """
    dataset = build_dataset()
    scorers = build_scorers()

    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=scorers,
        name=f"luau-eval-iter{iteration}-v{version}",
    )

    results = await evaluation.evaluate(generator)

    # Generate per-task outputs and score them individually for example collection
    per_task_details = []
    syntax_scorer = SyntaxScorer()
    api_scorer = ApiScorer()
    bug_scorer = BugScorer()
    quality_scorer = QualityScorer()

    for task in TEST_TASKS:
        try:
            code = generator.predict(task["task_description"])
        except Exception:
            code = ""

        # Score individually
        s_syntax = syntax_scorer.score(code)
        s_api = api_scorer.score(code)
        s_bug = bug_scorer.score(code)
        s_quality = quality_scorer.score(code)

        # Collect all issues
        all_issues = []
        all_issues.extend(s_syntax.get("syntax_issues", []))
        all_issues.extend(s_api.get("api_issues", []))
        for bug in s_bug.get("bugs_found", []):
            all_issues.append(bug["bug"])

        avg_score = (
            s_syntax.get("syntax_score", 0)
            + s_api.get("api_score", 0)
            + s_bug.get("bug_score", 0)
            + s_quality.get("quality_score", 0)
        ) / 4.0

        per_task_details.append({
            "task": task["task_description"],
            "category": task["category"],
            "code": code,
            "avg_score": avg_score,
            "issues": all_issues,
            "scores": {
                "syntax": s_syntax.get("syntax_score", 0),
                "api": s_api.get("api_score", 0),
                "bugs": s_bug.get("bug_score", 0),
                "quality": s_quality.get("quality_score", 0),
            },
        })

    # Sort by score ascending — worst first
    per_task_details.sort(key=lambda x: x["avg_score"])

    return results, per_task_details
