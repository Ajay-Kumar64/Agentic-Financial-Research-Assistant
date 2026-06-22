import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()  # Load API key from .env

class GeminiModel:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found! Add it to .env")
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-3-flash-preview"

    async def generate(self, query: str, docs: list[dict]) -> str:
        # Format context for citations
        context_blocks = []
        for i, d in enumerate(docs):
            source_info = f"Source: {d.get('filename','Unknown')} | Page: {d.get('page','N/A')}"
            context_blocks.append(f"[{i+1}] {source_info}\nContent: {d.get('text','')}")

        context_text = "\n\n".join(context_blocks)

        prompt = f"""You are a financial analyst. Answer the question using the provided sources.
CRITICAL: Use inline citations like [1], [2] to link your statements to the sources.
If the answer is not in the sources, say "I don't know."

SOURCES:
{context_text}

QUESTION: {query}
"""
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            return response.text.strip()
        except Exception as e:
            return f"LLM Generation Error: {str(e)}"
