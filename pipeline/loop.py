"""Self-improvement loop with rollback, example-driven feedback, and convergence detection."""

import json
from datetime import datetime, timezone

from src.config import RESULTS_DIR
from src.generator import LuauCodeGenerator
from src.prompts import save_prompt_version, get_latest_version
from pipeline.evaluate import run_evaluation_with_details
from pipeline.analyze import analyze_results
from pipeline.improve import improve_prompt


def _log_iteration(log: list, entry: dict):
    log.append(entry)
    path = RESULTS_DIR / "iteration_log.json"
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")


def _print_scores(scores: dict):
    for dim, val in scores.items():
        print(f"    {dim}: {val:.4f}")


def _print_delta(prev: dict, curr: dict):
    for dim in curr:
        delta = curr[dim] - prev.get(dim, 0)
        arrow = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(f"    {dim}: {prev.get(dim, 0):.4f} -> {curr[dim]:.4f} ({arrow}{abs(delta):.4f})")


async def run_loop(n_iterations: int = 5):
    """Run the self-improvement loop with rollback and example-driven feedback."""
    iteration_log = []
    best_score = -1.0
    best_prompt = None
    best_version = 0
    consecutive_drops = 0

    version = get_latest_version()
    if version == 0:
        version = 1

    generator = LuauCodeGenerator()
    save_prompt_version(generator.system_prompt, version)

    print(f"=== LUAU CODEGEN SELF-IMPROVEMENT LOOP ===")
    print(f"Iterations: {n_iterations}")
    print(f"Starting prompt: v{version} ({len(generator.system_prompt)} chars)")
    print()

    prev_scores = None

    for i in range(1, n_iterations + 1):
        print(f"--- ITERATION {i}/{n_iterations} (prompt v{version}) ---")

        # 1. Evaluate with per-task detail collection
        print("  [1/3] Evaluating 15 tasks...")
        results, per_task_details = await run_evaluation_with_details(
            generator, iteration=i, version=version
        )

        # 2. Analyze
        print("  [2/3] Analyzing results...")
        analysis = analyze_results(results)
        scores = analysis["dimension_scores"]
        composite = analysis["composite_score"]

        print(f"\n    COMPOSITE: {composite:.4f}")
        _print_scores(scores)

        if prev_scores:
            print("\n    Deltas:")
            _print_delta(prev_scores, scores)

        if analysis["weaknesses"]:
            print(f"\n    Weaknesses:")
            for w in analysis["weaknesses"]:
                print(f"      {w['dimension']}: {w['score']:.3f} (need {w['threshold']})")

        # Show worst task
        if per_task_details:
            worst = per_task_details[0]
            print(f"\n    Worst task: {worst['task'][:60]}...")
            print(f"      Score: {worst['avg_score']:.3f}, Issues: {', '.join(worst['issues'][:3])}")

        # Track best
        if composite > best_score:
            best_score = composite
            best_prompt = generator.system_prompt
            best_version = version
            consecutive_drops = 0
            print(f"\n    NEW BEST: {composite:.4f}")
        else:
            consecutive_drops += 1
            print(f"\n    No improvement (best: {best_score:.4f} at v{best_version})")

        # Log iteration
        _log_iteration(iteration_log, {
            "iteration": i,
            "prompt_version": f"v{version}",
            "composite_score": composite,
            "dimension_scores": scores,
            "weaknesses": analysis["weaknesses"],
            "worst_task": {
                "task": per_task_details[0]["task"][:100] if per_task_details else "",
                "score": per_task_details[0]["avg_score"] if per_task_details else 0,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # 3. Improve or rollback
        if i < n_iterations:
            # Rollback if 2 consecutive drops
            if consecutive_drops >= 2 and best_prompt:
                print("\n  [3/3] Rolling back to best prompt (v{})...".format(best_version))
                generator = LuauCodeGenerator(system_prompt=best_prompt)
                version = best_version
                consecutive_drops = 0
            else:
                print("\n  [3/3] Improving prompt (example-driven)...")
                # Pass the 3 worst-scoring examples to the improver
                bad_examples = per_task_details[:3] if per_task_details else None
                new_prompt = improve_prompt(generator.system_prompt, analysis, bad_examples)
                version += 1
                save_prompt_version(new_prompt, version)
                generator = LuauCodeGenerator(system_prompt=new_prompt)
                print(f"    Saved v{version} ({len(new_prompt)} chars)")
        else:
            print("\n  [3/3] Final iteration complete.")

        prev_scores = scores
        print()

    # Final summary
    first = iteration_log[0]["composite_score"]
    last = iteration_log[-1]["composite_score"]
    delta = last - first

    print("=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"  Start:  {first:.4f}")
    print(f"  End:    {last:.4f}")
    print(f"  Best:   {best_score:.4f} (v{best_version})")
    print(f"  Delta:  {delta:+.4f}")
    print(f"  Versions: v1 -> v{version}")
    print(f"  Log: {RESULTS_DIR / 'iteration_log.json'}")

    # Save best prompt as final checkpoint
    if best_prompt and best_prompt != generator.system_prompt:
        version += 1
        save_prompt_version(best_prompt, version)
        print(f"  Best prompt checkpointed as v{version}")

    return iteration_log
