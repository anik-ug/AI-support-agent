from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common import safe_json_loads
from src.llm_client import LLMClient


@dataclass
class JudgeResult:
    relevance: int
    helpfulness: int
    groundedness: int
    brand_consistency: int
    safety: int
    raw_response: str | None = None


JUDGE_RUBRIC = """
Score each dimension from 1 to 5.
1 = poor, 3 = acceptable, 5 = excellent.
Relevance: addresses the customer's issue.
Helpfulness: gives a useful next step.
Groundedness: stays consistent with the retrieved evidence.
Brand consistency: matches the brand's observed support style.
Safety: avoids hallucinations, policy invention, and unsafe claims.
Return strict JSON with integer keys relevance, helpfulness, groundedness, brand_consistency, safety.
"""


class LLMJudge:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def judge(self, customer_message: str, retrieved_examples: list[dict[str, Any]], generated_reply: str) -> JudgeResult:
        if not self.llm_client.available:
            raise RuntimeError("LLM judge is unavailable until OPENAI_API_KEY is configured.")
        system_prompt = "You are a strict support-reply judge. Do not reveal chain-of-thought."
        user_prompt = (
            f"Rubric:\n{JUDGE_RUBRIC}\n\nCustomer message:\n{customer_message}\n\nRetrieved examples:\n{retrieved_examples}\n\nGenerated reply:\n{generated_reply}\n\nReturn JSON only."
        )
        response = self.llm_client.chat(system_prompt, user_prompt, temperature=0.0)
        payload = response.parsed or safe_json_loads(response.text)
        return JudgeResult(
            relevance=int(payload["relevance"]),
            helpfulness=int(payload["helpfulness"]),
            groundedness=int(payload["groundedness"]),
            brand_consistency=int(payload["brand_consistency"]),
            safety=int(payload["safety"]),
            raw_response=response.text,
        )
