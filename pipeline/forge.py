"""The Code Forge: Self-Correcting, Self-Evolving Code Generation Pipeline.

Orchestrates the full loop:
1. Generate code for tasks
2. Diagnose issues with scorers
3. Self-heal (model fixes its own code)
4. Learn from corrections (create training data)
5. Evolve scorers (discover new patterns)
6. Generate harder tasks (adversarial curriculum)
7. Repeat
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import RESULTS_DIR
from src.generator import LuauCodeGenerator
from src.doctor import diagnose, self_correct
from src.curriculum import escalate_difficulty
from src.self_evolve import (
    analyze_correction,
    discover_new_patterns,
    generate_scorer_rule,
    get_evolved_rules,
    apply_evolved_rules,
)
from data.test_tasks import TEST_TASKS


def _save_log(log: list, path: Path):
    path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")


async def run_forge(
    n_rounds: int = 3,
    max_heal_attempts: int = 3,
    tasks: list = None,
) -> dict:
    """Run the full Code Forge pipeline.

    Args:
        n_rounds: Number of forge rounds (each escalates difficulty)
        max_heal_attempts: Max self-correction attempts per task
        tasks: Override tasks (default: TEST_TASKS)

    Returns:
        Complete forge results with metrics and correction history.
    """
    generator = LuauCodeGenerator()
    forge_log = []
    all_corrections = []
    current_tasks = tasks or TEST_TASKS
    difficulty = 0

    print("=== CODE FORGE: SELF-CORRECTING PIPELINE ===")
    print(f"Rounds: {n_rounds}, Tasks: {len(current_tasks)}")
    print()

    for round_num in range(1, n_rounds + 1):
        print(f"--- ROUND {round_num}/{n_rounds} (difficulty {difficulty}) ---")
        round_results = []
        round_corrections = []

        for i, task in enumerate(current_tasks):
            task_desc = task["task_description"]
            category = task.get("category", "unknown")
            print(f"  [{i+1}/{len(current_tasks)}] {task_desc[:60]}...")

            # 1. Generate
            try:
                code = generator.predict(task_desc)
            except Exception as e:
                print(f"    Generation failed: {e}")
                continue

            # 2. Diagnose + Self-correct
            result = self_correct(
                task_desc, code, max_rounds=max_heal_attempts
            )

            # 3. Apply evolved rules to final code
            evolved_issues = apply_evolved_rules(result["final_code"])
            if evolved_issues:
                result["evolved_issues"] = evolved_issues

            print(
                f"    Score: {result['original_score']:.2f} -> "
                f"{result['final_score']:.2f} "
                f"({'PASS' if result['passed'] else 'FAIL'}) "
                f"[{result['rounds']} rounds]"
            )

            round_results.append({
                "task": task_desc,
                "category": category,
                "original_score": result["original_score"],
                "final_score": result["final_score"],
                "improvement": result["final_score"] - result["original_score"],
                "passed": result["passed"],
                "heal_rounds": result["rounds"],
            })

            if result["improved"]:
                round_corrections.append({
                    "original_code": result["original_code"],
                    "final_code": result["final_code"],
                    "task": task_desc,
                    "category": category,
                })

        # Round summary
        avg_original = sum(r["original_score"] for r in round_results) / max(len(round_results), 1)
        avg_final = sum(r["final_score"] for r in round_results) / max(len(round_results), 1)
        pass_rate = sum(1 for r in round_results if r["passed"]) / max(len(round_results), 1)

        print(f"\n  Round {round_num} summary:")
        print(f"    Avg score: {avg_original:.3f} -> {avg_final:.3f} (+{avg_final - avg_original:.3f})")
        print(f"    Pass rate: {pass_rate:.0%}")
        print(f"    Corrections: {len(round_corrections)}")

        # 4. Evolve scorers from this round's corrections
        if round_corrections:
            all_corrections.extend(round_corrections)
            new_patterns = discover_new_patterns(round_corrections)
            if new_patterns:
                print(f"    New patterns discovered: {len(new_patterns)}")
                for p in new_patterns:
                    rule = generate_scorer_rule(p)
                    if rule:
                        print(f"      + {rule['description']}")

        # 5. Build category scores for curriculum
        category_scores = {}
        for r in round_results:
            cat = r["category"]
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(r["final_score"])

        category_avgs = {
            cat: sum(scores) / len(scores)
            for cat, scores in category_scores.items()
        }

        # Log round
        forge_log.append({
            "round": round_num,
            "difficulty": difficulty,
            "n_tasks": len(current_tasks),
            "avg_original_score": round(avg_original, 3),
            "avg_final_score": round(avg_final, 3),
            "pass_rate": round(pass_rate, 3),
            "corrections_made": len(round_corrections),
            "evolved_rules": len(get_evolved_rules()),
            "category_scores": {k: round(v, 3) for k, v in category_avgs.items()},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # 6. Escalate curriculum for next round
        if round_num < n_rounds:
            print(f"\n  Escalating curriculum...")
            try:
                harder_tasks, difficulty = escalate_difficulty(
                    {"category_scores": category_avgs},
                    current_level=difficulty,
                )
                # Mix: keep some original tasks + add harder ones
                current_tasks = current_tasks[:10] + harder_tasks
                print(f"    Added {len(harder_tasks)} harder tasks (difficulty {difficulty})")
            except Exception as e:
                print(f"    Curriculum escalation failed: {e}")

        print()

    # Save results
    log_path = RESULTS_DIR / "forge_log.json"
    _save_log(forge_log, log_path)

    corrections_path = RESULTS_DIR / "forge_corrections.json"
    correction_data = []
    for c in all_corrections:
        correction_data.append({
            "task": c["task"],
            "category": c["category"],
            "original": c["original_code"][:500],
            "corrected": c["final_code"][:500],
        })
    _save_log(correction_data, corrections_path)

    # Final summary
    first_avg = forge_log[0]["avg_original_score"]
    last_avg = forge_log[-1]["avg_final_score"]
    total_rules = len(get_evolved_rules())

    print("=" * 50)
    print("FORGE COMPLETE")
    print("=" * 50)
    print(f"  Rounds: {n_rounds}")
    print(f"  Score: {first_avg:.3f} -> {last_avg:.3f} (+{last_avg - first_avg:.3f})")
    print(f"  Total corrections: {len(all_corrections)}")
    print(f"  Evolved rules: {total_rules}")
    print(f"  Log: {log_path}")

    return {
        "forge_log": forge_log,
        "corrections": len(all_corrections),
        "evolved_rules": total_rules,
        "start_score": first_avg,
        "end_score": last_avg,
        "improvement": last_avg - first_avg,
    }
