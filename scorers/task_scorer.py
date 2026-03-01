import json
import re
import weave
from src.mistral_client import generate_completion


JUDGE_SYSTEM_PROMPT = """\
You are an expert Roblox Luau code reviewer. You will evaluate generated code against a task description.

Score each dimension from 0.0 to 1.0:
- functionality: Does the code implement the requested features?
- correctness: Is the code logically correct and free of errors?
- completeness: Does the code handle all aspects of the task, including edge cases?

Respond with ONLY a JSON object in this exact format:
{
    "functionality": 0.0,
    "correctness": 0.0,
    "completeness": 0.0,
    "reasoning": "Brief explanation"
}\
"""


class TaskCompletionScorer(weave.Scorer):
    """Uses Mistral as an LLM-judge to evaluate task completion."""

    @weave.op()
    def score(self, output: str, task_description: str, **kwargs) -> dict:
        user_prompt = (
            f"## Task Description\n{task_description}\n\n"
            f"## Generated Code\n```lua\n{output}\n```\n\n"
            "Evaluate how well the code fulfills the task. Respond with JSON only."
        )

        raw = generate_completion(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=512,
        )

        parsed = self._parse_response(raw)

        # Weighted composite: functionality 0.4, correctness 0.35, completeness 0.25
        composite = (
            parsed["functionality"] * 0.4
            + parsed["correctness"] * 0.35
            + parsed["completeness"] * 0.25
        )

        return {
            "task_completion_score": round(composite, 3),
            "functionality": parsed["functionality"],
            "correctness": parsed["correctness"],
            "completeness": parsed["completeness"],
            "judge_reasoning": parsed["reasoning"],
        }

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Extract JSON from the LLM response, handling markdown fences."""
        # Try to extract JSON from markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            raw = match.group(1)
        else:
            # Try raw JSON
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)

        try:
            data = json.loads(raw)
            return {
                "functionality": float(data.get("functionality", 0.5)),
                "correctness": float(data.get("correctness", 0.5)),
                "completeness": float(data.get("completeness", 0.5)),
                "reasoning": str(data.get("reasoning", "No reasoning provided")),
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "functionality": 0.5,
                "correctness": 0.5,
                "completeness": 0.5,
                "reasoning": f"Failed to parse judge response: {raw[:200]}",
            }
