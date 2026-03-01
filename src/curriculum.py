"""Adversarial Curriculum Generator.

Analyzes model weaknesses and generates progressively harder tasks
that specifically target failure patterns. Each round increases difficulty.
"""

from src.mistral_client import generate_completion


# Difficulty templates per category
DIFFICULTY_MODIFIERS = {
    "data_persistence": [
        "with retry logic on DataStore failures",
        "with session locking to prevent data corruption from multiple servers",
        "with data versioning, migration from old schemas, and ordered DataStore leaderboards",
    ],
    "remote_events": [
        "with basic client-server validation",
        "with rate limiting, type checking, and anti-exploit validation on all RemoteEvents",
        "with encrypted payloads, server-authoritative state, and replay attack prevention",
    ],
    "npc_behavior": [
        "with basic pathfinding to nearest player",
        "with state machine (idle/chase/attack/flee), obstacle avoidance, and target switching",
        "with group coordination, flanking behavior, and dynamic difficulty scaling based on player count",
    ],
    "ui": [
        "with a toggle button and basic layout",
        "with animated transitions, drag-and-drop, and responsive scaling across devices",
        "with data binding, virtual scrolling for 1000+ items, and accessibility support",
    ],
    "game_mechanics": [
        "with basic collision detection",
        "with cooldowns, combo chains, and knockback physics using CFrame math",
        "with server-authoritative hit detection, lag compensation, and anti-cheat validation",
    ],
    "physics_cframe": [
        "with basic CFrame positioning",
        "with smooth interpolation using CFrame:Lerp and Bezier curves",
        "with inverse kinematics, constraint-based physics, and custom collision resolution",
    ],
}


def generate_harder_tasks(
    weak_categories: list[dict],
    difficulty_level: int = 1,
    n_tasks: int = 5,
) -> list[dict]:
    """Generate harder task variants targeting weak categories.

    Args:
        weak_categories: List of {"category": str, "score": float}
        difficulty_level: 0=easy, 1=medium, 2=hard
        n_tasks: Number of tasks to generate

    Returns:
        List of {"task_description": str, "category": str, "difficulty": int}
    """
    tasks = []
    level = min(difficulty_level, 2)

    for weakness in weak_categories:
        cat = weakness["category"]
        modifiers = DIFFICULTY_MODIFIERS.get(cat, DIFFICULTY_MODIFIERS["game_mechanics"])
        modifier = modifiers[level]

        prompt = f"""Generate a specific Roblox Luau coding task for category: {cat}

The task must be difficulty level {level + 1}/3 and must include: {modifier}

Write a 2-3 sentence task description that a programmer would receive.
Be very specific about what the script should do.
Output ONLY the task description."""

        system = (
            "You are a Roblox game development instructor. "
            "Generate specific, detailed coding tasks."
        )

        desc = generate_completion(system, prompt, temperature=0.7)
        tasks.append({
            "task_description": desc.strip(),
            "category": cat,
            "difficulty": level + 1,
        })

        if len(tasks) >= n_tasks:
            break

    return tasks


def escalate_difficulty(eval_results: dict, current_level: int = 0) -> tuple[list[dict], int]:
    """Analyze eval results and escalate difficulty for weak areas.

    Returns (new_tasks, new_difficulty_level).
    """
    # Find weak categories
    weak = []
    category_scores = eval_results.get("category_scores", {})

    for cat, score in category_scores.items():
        if score < 0.80:
            weak.append({"category": cat, "score": score})

    # Sort by weakness (lowest first)
    weak.sort(key=lambda x: x["score"])

    if not weak:
        # Everything passing — escalate difficulty
        new_level = min(current_level + 1, 2)
        # Generate general harder tasks
        weak = [{"category": cat, "score": 0.75} for cat in list(DIFFICULTY_MODIFIERS.keys())[:3]]
    else:
        new_level = current_level

    tasks = generate_harder_tasks(weak, difficulty_level=new_level, n_tasks=5)
    return tasks, new_level
