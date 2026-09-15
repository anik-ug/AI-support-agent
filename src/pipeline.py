from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.common import stable_hash
from src.escalation import decide_escalation
from src.intent_classifier import IntentPrediction
from src.llm_client import LLMClient
from src.reply_generator import GroundedReplyGenerator
from src.retriever import RetrievedExample, SentenceTransformerRetriever


@dataclass
class SupportAgentResult:
    intent: str
    intent_confidence: float
    retrieved_examples: list[dict[str, Any]]
    reply: str
    action: str
    reason: str
    llm_fallback_used: bool
    llm_warning: str | None


class SupportAgent:
    def __init__(
        self,
        *,
        config: AppConfig,
        taxonomy: dict[str, Any],
        retriever: SentenceTransformerRetriever,
        intent_model: Any,
        reply_generator: GroundedReplyGenerator,
    ):
        self.config = config
        self.taxonomy = taxonomy
        self.allowed_intents = [intent["name"] for intent in taxonomy.get("intents", [])]
        self.retriever = retriever
        self.intent_model = intent_model
        self.reply_generator = reply_generator

    @classmethod
    def from_artifacts(cls, config: AppConfig, taxonomy: dict[str, Any], brand_name: str, retriever_path: Path | None = None, intent_model: Any | None = None) -> "SupportAgent":
        llm_client = LLMClient(provider=config.llm_provider, model=config.llm_model, api_key=(
            config.openai_api_key
            if config.llm_provider == "openai"
            else (config.gemini_api_key or config.llm_api_key)
        ))
        retriever = SentenceTransformerRetriever.load(retriever_path) if retriever_path else SentenceTransformerRetriever(config.embedding_model)
        reply_generator = GroundedReplyGenerator(llm_client=llm_client, brand_name=brand_name)
        return cls(config=config, taxonomy=taxonomy, retriever=retriever, intent_model=intent_model, reply_generator=reply_generator)

    def run(self, message: str) -> SupportAgentResult:
        intent_prediction: IntentPrediction = self.intent_model.predict_one(message)
        retrieved: list[RetrievedExample] = self.retriever.search(
            message,
            top_k=self.config.top_k_retrieval,
            exclude_hashes={stable_hash(message)},
        )
        top_similarity = retrieved[0].score if retrieved else 0.0
        decision = decide_escalation(
            intent=intent_prediction.intent,
            intent_confidence=float(intent_prediction.confidence),
            top_similarity=float(top_similarity),
            retrieved_count=len(retrieved),
            text=message,
            allowed_intents=self.allowed_intents,
            min_intent_confidence=self.config.escalation.min_intent_confidence,
            min_similarity=self.config.escalation.min_similarity,
            min_retrieval_hits=self.config.escalation.min_retrieval_hits,
        )
        reply_result = self.reply_generator.generate(message, intent_prediction.intent, retrieved) if decision.action == "AUTO_HANDLE" else self.reply_generator.fallback_reply(retrieved)
        llm_warning = None
        if intent_prediction.used_fallback:
            llm_warning = "LLM fallback used for intent classification because the live model call was unavailable or quota-limited."
        return SupportAgentResult(
            intent=intent_prediction.intent,
            intent_confidence=float(intent_prediction.confidence),
            retrieved_examples=[example.__dict__ for example in retrieved],
            reply=reply_result.reply,
            action=decision.action,
            reason=decision.reason,
            llm_fallback_used=bool(intent_prediction.used_fallback),
            llm_warning=llm_warning,
        )
