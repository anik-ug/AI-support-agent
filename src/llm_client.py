from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common import safe_json_loads

@dataclass
class LLMResponse:
    text: str
    parsed: Any | None = None

class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str | None):
        self.provider = provider.lower().strip()
        self.model = model
        self.api_key = api_key
        self._client = None
        if not api_key:
            return
        try:
            from openai import OpenAI
            if self.provider == "gemini":
                self._client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            else:
                self._client = OpenAI(api_key=api_key)
        except Exception:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        if not self.available:
            raise RuntimeError("LLM client is unavailable. Check LLM_PROVIDER and API key in .env")
        completion = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = completion.choices[0].message.content or ""
        parsed = None
        try:
            parsed = safe_json_loads(text)
        except Exception:
            pass
        return LLMResponse(text=text, parsed=parsed)
