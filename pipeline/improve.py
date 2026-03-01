"""Example-driven prompt improvement — shows the improver actual failing code."""

import weave
from src.mistral_client import generate_completion


DIMENSION_FIX_GUIDES = {
    "syntax": (
        "The generated code has syntax issues like mismatched blocks or brackets. "
        "Add a clear rule: 'Count your function/if/for/while/do blocks and ensure each has a matching end. "
        "Never output partial code.' Keep it concise — one or two sentences max."
    ),
    "api": (
        "The code uses deprecated Roblox APIs. Strengthen the rule about using task.wait() over wait(), "
        "game:GetService() over direct access, and .Connect() over .connect(). "
        "Add a short list of the 3 most common mistakes to avoid."
    ),
    "bugs": (
        "The code has bug patterns: unchecked FindFirstChild, missing pcall on DataStore, "
        "or infinite loops without yields. Add specific guard patterns as one-line examples, e.g. "
        "'local obj = parent:FindFirstChild(name); if not obj then return end'"
    ),
    "quality": (
        "Code quality is low: missing comments, poor organization, or bad naming. "
        "Add a rule like: 'Start with a 1-line comment describing the script. "
        "Group code into sections: Services, Config, Functions, Init. "
        "Use PascalCase for services, camelCase for locals.'"
    ),
    "task": (
        "The code doesn't fully implement the requested features. "
        "Add: 'Read the task description carefully. Implement every feature mentioned. "
        "Never leave TODO comments or placeholder functions. Every function must have a complete body.'"
    ),
}


META_PROMPT = """\
You are rewriting a system prompt for a Roblox Luau code generator. The current prompt produces code that scores poorly in some areas.

CURRENT PROMPT:
---
{current_prompt}
---

SCORES (0.0-1.0, higher=better):
{scores_text}

WEAKEST AREAS AND HOW TO FIX:
{fix_instructions}

EXAMPLES OF PROBLEMATIC OUTPUT (from the worst-scoring generations):
{examples}

RULES FOR THE REWRITE:
1. Keep the prompt under 40 lines. Shorter prompts produce better code than verbose ones.
2. Keep every rule that's working well (high-scoring dimensions).
3. For weak dimensions, add ONE specific, actionable rule with a short code example.
4. Don't use emphatic language (NEVER, MUST, ALWAYS in caps) — just state the rule clearly.
5. End with: "Output only the Luau code, no markdown fences or explanations."
6. Don't add meta-instructions like "double-check your code" — the model can't do that.

Output ONLY the new system prompt. No explanations, no markdown fences.\
"""


@weave.op()
def improve_prompt(
    current_prompt: str,
    analysis: dict,
    bad_examples: list[dict] | None = None,
) -> str:
    """Generate an improved system prompt using weakness analysis and failing examples."""
    scores = analysis["dimension_scores"]
    weaknesses = analysis["weaknesses"]

    scores_text = "\n".join(f"  {dim}: {score:.3f}" for dim, score in scores.items())

    # Build targeted fix instructions
    fix_lines = []
    target_dims = [w["dimension"] for w in weaknesses] if weaknesses else []
    # Always include the 2 worst dimensions even if above threshold
    if len(target_dims) < 2:
        sorted_dims = sorted(scores.items(), key=lambda x: x[1])
        for dim, _ in sorted_dims:
            if dim not in target_dims:
                target_dims.append(dim)
            if len(target_dims) >= 2:
                break

    for dim in target_dims:
        guide = DIMENSION_FIX_GUIDES.get(dim, "")
        score = scores.get(dim, 0)
        fix_lines.append(f"- {dim} (score: {score:.3f}): {guide}")

    fix_instructions = "\n".join(fix_lines)

    # Format bad examples (truncated to avoid token bloat)
    examples_text = "None available."
    if bad_examples:
        example_parts = []
        for i, ex in enumerate(bad_examples[:3]):  # Max 3 examples
            task = ex.get("task", "Unknown task")
            code = ex.get("code", "")
            issues = ex.get("issues", [])
            # Truncate code to first 40 lines
            code_lines = code.split("\n")[:40]
            truncated = "\n".join(code_lines)
            if len(code_lines) < len(code.split("\n")):
                truncated += "\n... (truncated)"
            issues_str = "; ".join(issues[:5]) if issues else "Low overall score"
            example_parts.append(
                f"Example {i+1} — Task: {task[:80]}\n"
                f"Issues: {issues_str}\n"
                f"Code (first 40 lines):\n```lua\n{truncated}\n```"
            )
        examples_text = "\n\n".join(example_parts)

    prompt = META_PROMPT.format(
        current_prompt=current_prompt,
        scores_text=scores_text,
        fix_instructions=fix_instructions,
        examples=examples_text,
    )

    new_prompt = generate_completion(
        system_prompt="You are a prompt engineering expert. Output only the improved system prompt text.",
        user_prompt=prompt,
        temperature=0.3,
        max_tokens=3000,
    )

    # Clean markdown fences if present
    new_prompt = new_prompt.strip()
    if new_prompt.startswith("```"):
        lines = new_prompt.split("\n")
        new_prompt = "\n".join(lines[1:])
    if new_prompt.endswith("```"):
        new_prompt = new_prompt[:-3].rstrip()

    return new_prompt
