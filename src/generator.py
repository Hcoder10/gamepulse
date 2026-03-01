import re
import weave
from src.mistral_client import generate_completion
from src.prompts import DEFAULT_SYSTEM_PROMPT


class LuauCodeGenerator(weave.Model):
    """Generates Roblox Luau code from task descriptions.

    Every field change (including system_prompt) creates a new Weave model version.
    """
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model_name: str = "mistral-large-latest"
    temperature: float = 0.7

    @weave.op()
    def predict(self, task_description: str) -> str:
        """Generate Luau code for the given task description."""
        raw = generate_completion(
            system_prompt=self.system_prompt,
            user_prompt=task_description,
            model=self.model_name,
            temperature=self.temperature,
        )
        return self._strip_markdown_fences(raw)

    @staticmethod
    def _strip_markdown_fences(code: str) -> str:
        """Remove markdown code fences if present."""
        code = code.strip()
        code = re.sub(r"^```(?:lua|luau)?\s*\n?", "", code)
        code = re.sub(r"\n?```\s*$", "", code)
        return code.strip()
