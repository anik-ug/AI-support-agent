from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common import safe_json_loads
from src.llm_client import LLMClient
from src.retriever import RetrievedExample


@dataclass
class ReplyResult:
    reply: str
    grounding_examples: list[dict[str, Any]]
    raw_response: str | None = None


def build_reply_prompt(
    *,
    customer_message: str,
    intent: str,
    retrieved_examples: list[RetrievedExample],
    brand_name: str,
) -> tuple[str, str]:
    examples_text = []
    for idx, example in enumerate(retrieved_examples, start=1):
        examples_text.append(
            f"Example {idx}:\nCustomer: {example.message}\nBrand reply: {example.brand_reply}\nSimilarity: {example.score:.3f}"
        )
    system_prompt = (
        "You are a customer-support drafting assistant. Use the retrieved historical examples as evidence. "
        "Follow the selected brand's historical support behavior. Do not invent policies, refunds, credits, timelines, or account actions. "
        "If evidence is insufficient, prefer escalation. Keep the reply concise and professional. "
        "Do not blindly copy historical replies. Return strict JSON with keys reply and grounding_examples."
    )
    user_prompt = (
        f"Brand: {brand_name}\nIntent: {intent}\nCustomer message: {customer_message}\n\nRetrieved historical examples:\n"
        + "\n\n".join(examples_text)
        + "\n\nReturn JSON only."
    )
    return system_prompt, user_prompt


class GroundedReplyGenerator:
    def __init__(self, llm_client: LLMClient, brand_name: str):
        self.llm_client = llm_client
        self.brand_name = brand_name

    def fallback_reply(self, retrieved_examples: list[RetrievedExample]) -> ReplyResult:
        if retrieved_examples:
            reply = "Thanks for reaching out. We found similar support cases, but a support specialist should review the request before a final action is taken." if retrieved_examples else "Thanks for reaching out. A support specialist should review this request."
        return ReplyResult(reply=reply, grounding_examples=[example.__dict__ for example in retrieved_examples[:3]], raw_response=None)

    def generate(self, customer_message: str, intent: str, retrieved_examples: list[RetrievedExample]) -> ReplyResult:
        if not self.llm_client.available:
            return self.fallback_reply(retrieved_examples)
        system_prompt, user_prompt = build_reply_prompt(
            customer_message=customer_message,
            intent=intent,
            retrieved_examples=retrieved_examples,
            brand_name=self.brand_name,
        )
        try:
            response = self.llm_client.chat(system_prompt, user_prompt, temperature=0.2)
            payload = response.parsed or safe_json_loads(response.text)
            reply = str(payload["reply"])
            grounding = payload.get("grounding_examples", [])
            if not isinstance(grounding, list):
                grounding = []
            return ReplyResult(reply=reply, grounding_examples=grounding, raw_response=response.text)
        except Exception:
            return self.fallback_reply(retrieved_examples)
