import time
import weave
from mistralai import Mistral
from src.config import MISTRAL_API_KEY, MISTRAL_MODEL


_client = Mistral(api_key=MISTRAL_API_KEY)

MAX_RETRIES = 3
BASE_DELAY = 2.0


@weave.op()
def generate_completion(
    system_prompt: str,
    user_prompt: str,
    model: str = MISTRAL_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Call Mistral chat API with retry and exponential backoff."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = _client.chat.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = BASE_DELAY * (2 ** attempt)
            print(f"  Retry {attempt + 1}/{MAX_RETRIES} after {delay}s: {e}")
            time.sleep(delay)
