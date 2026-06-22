import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

_client = None
_model_id = "gemini-3.1-flash-lite"

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set in environment")
        _client = genai.Client(api_key=api_key)
    return _client


def call_llm_sync(prompt: str, system_instruction: str = "", temperature: float = 0.0):
    """
    Synchronous LLM call with exponential backoff for rate limits.
    """
    client = _get_client()
    config = types.GenerateContentConfig(temperature=temperature)
    if system_instruction:
        config.system_instruction = system_instruction

    # Exponential backoff for rate limits
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=_model_id,
                contents=prompt,
                config=config
            )
            text = response.text if hasattr(response, "text") else str(response)

            tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                tokens = getattr(um, "prompt_token_count", 0) + getattr(um, "candidates_token_count", 0)
            else:
                tokens = len(prompt.split()) + len(text.split())

            return text, tokens

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = (2 ** attempt) + 1  # 3, 6, 12, 36 seconds
                print(f"[LLM] Rate limited (429). Retrying in {wait}s... (attempt {attempt + 1}/4)")
                time.sleep(wait)
            else:
                print(f"[LLM] Error: {error_str}")
                return f"Error: {error_str}", 0

    return "Error: Max retries exceeded for rate limit", 0
# =============================================================================
# WARM-UP: Pay the 11s cold-start cost at module import, not on first query
# =============================================================================
print(f"[LLM] Warming up {_model_id}...")
t0 = time.time()
_, _ = call_llm_sync("warmup", system_instruction="You are a planner.", temperature=0.0)
print(f"[LLM] ✅ Ready in {time.time() - t0:.2f}s")