"""Self-Evolving Scorer: Discovers new patterns and adds rules.

Analyzes the corrections made during self-healing to discover
what patterns the model keeps getting wrong, then generates
new scorer rules to catch those patterns.
"""

import re
from src.mistral_client import generate_completion


# Dynamic rules discovered during self-correction
DISCOVERED_RULES = []


def analyze_correction(original: str, corrected: str) -> list[str]:
    """Analyze what changed between original and corrected code.

    Returns list of pattern descriptions that were fixed.
    """
    changes = []

    # Check what was added in the correction
    orig_lines = set(original.strip().split("\n"))
    corr_lines = set(corrected.strip().split("\n"))
    added = corr_lines - orig_lines
    removed = orig_lines - corr_lines

    added_text = "\n".join(added)
    removed_text = "\n".join(removed)

    # Detect common fix patterns
    if "pcall" in added_text and "pcall" not in original:
        changes.append("Added pcall wrapping for error handling")

    if "task.wait" in added_text and "task.wait" not in original:
        changes.append("Added yield (task.wait) to prevent infinite loop")

    if "FindFirstChild" in added_text or ":WaitForChild" in added_text:
        changes.append("Added nil-safety checks before accessing children")

    if "Disconnect" in added_text and "Disconnect" not in original:
        changes.append("Added connection cleanup (Disconnect)")

    if "local " in added_text:
        # Check if globals were converted to locals
        new_locals = re.findall(r"\blocal\s+(\w+)", added_text)
        if new_locals:
            changes.append(f"Converted globals to locals: {', '.join(new_locals[:3])}")

    if ":GetService" in added_text and "game." in removed_text:
        changes.append("Replaced direct service access with GetService")

    if "task.spawn" in added_text and "spawn(" in removed_text:
        changes.append("Replaced deprecated spawn() with task.spawn()")

    if "task.wait" in added_text and re.search(r"\bwait\(", removed_text):
        changes.append("Replaced deprecated wait() with task.wait()")

    return changes


def discover_new_patterns(correction_history: list[dict]) -> list[dict]:
    """Analyze multiple corrections to discover recurring patterns.

    Returns list of new patterns that should become scorer rules.
    """
    all_changes = []
    for entry in correction_history:
        if "original_code" in entry and "final_code" in entry:
            changes = analyze_correction(
                entry["original_code"],
                entry["final_code"]
            )
            all_changes.extend(changes)

    # Count recurring patterns
    pattern_counts = {}
    for change in all_changes:
        # Normalize
        key = change.lower().strip()
        pattern_counts[key] = pattern_counts.get(key, 0) + 1

    # Patterns that appear 2+ times are worth adding as rules
    new_patterns = []
    for pattern, count in pattern_counts.items():
        if count >= 2:
            new_patterns.append({
                "pattern": pattern,
                "occurrences": count,
                "severity": "high" if count >= 4 else "medium",
            })

    return new_patterns


def generate_scorer_rule(pattern: dict) -> dict:
    """Generate a new scorer rule from a discovered pattern.

    Uses Mistral to create a regex pattern and description
    for the new rule.
    """
    prompt = f"""A code review found this recurring issue in Roblox Luau code:
"{pattern['pattern']}"
This appeared {pattern['occurrences']} times across multiple scripts.

Generate a Python regex pattern that would detect this issue in Luau code.
Respond in this exact format:
REGEX: <the regex pattern>
DESCRIPTION: <one line description>
SEVERITY: <high/medium/low>"""

    system = "You are a code analysis expert. Generate precise regex patterns."

    response = generate_completion(system, prompt, temperature=0.2)

    # Parse response
    regex_match = re.search(r"REGEX:\s*(.+)", response)
    desc_match = re.search(r"DESCRIPTION:\s*(.+)", response)
    severity_match = re.search(r"SEVERITY:\s*(\w+)", response)

    if regex_match:
        rule = {
            "regex": regex_match.group(1).strip(),
            "description": desc_match.group(1).strip() if desc_match else pattern["pattern"],
            "severity": severity_match.group(1).strip().lower() if severity_match else "medium",
            "source": "self-evolved",
            "occurrences": pattern["occurrences"],
        }

        # Validate regex
        try:
            re.compile(rule["regex"])
            DISCOVERED_RULES.append(rule)
            return rule
        except re.error:
            return None

    return None


def get_evolved_rules() -> list[dict]:
    """Return all dynamically discovered rules."""
    return DISCOVERED_RULES.copy()


def apply_evolved_rules(code: str) -> list[str]:
    """Apply all discovered rules to code and return issues found."""
    issues = []
    for rule in DISCOVERED_RULES:
        try:
            if re.search(rule["regex"], code):
                issues.append(f"[EVOLVED] {rule['description']}")
        except re.error:
            continue
    return issues
