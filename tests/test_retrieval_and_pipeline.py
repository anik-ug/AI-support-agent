from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.pipeline import SupportAgent
from src.reply_generator import GroundedReplyGenerator, ReplyResult
from src.retriever import RetrievedExample, SentenceTransformerRetriever


def test_retriever_returns_similar_examples(monkeypatch):
    def fake_encode(self, texts):
        vectors = []
        for text in texts:
            vectors.append([float("login" in text.lower()), float("error" in text.lower())])
        return np.array(vectors, dtype="float32")

    monkeypatch.setattr(SentenceTransformerRetriever, "encode", fake_encode, raising=False)
    df = pd.DataFrame(
        [
            {"customer_message": "login error", "brand_reply": "please reset", "conversation_id": "1", "brand_id": "brand_a", "customer_message_hash": "a", "brand_reply_hash": "b"},
            {"customer_message": "refund status", "brand_reply": "we are checking", "conversation_id": "2", "brand_id": "brand_a", "customer_message_hash": "c", "brand_reply_hash": "d"},
        ]
    )
    retriever = SentenceTransformerRetriever()
    retriever.fit(df)
    results = retriever.search("login error", top_k=1)
    assert results
    assert results[0].message == "login error"


def test_retriever_hashing_fallback_runs_without_embedding_model():
    df = pd.DataFrame(
        [
            {"customer_message": "cannot log in", "brand_reply": "reset your password", "conversation_id": "1", "brand_id": "brand_a", "customer_message_hash": "a", "brand_reply_hash": "b"},
            {"customer_message": "where is my refund", "brand_reply": "we are checking", "conversation_id": "2", "brand_id": "brand_a", "customer_message_hash": "c", "brand_reply_hash": "d"},
        ]
    )
    retriever = SentenceTransformerRetriever()
    retriever.backend = "hashing"
    retriever.fit(df)

    results = retriever.search("I cannot log in", top_k=1)

    assert retriever.dimension == 384
    assert results[0].message == "cannot log in"


class FakeIntentModel:
    def predict_one(self, message: str):
        from src.intent_classifier import IntentPrediction

        return IntentPrediction(intent="login_issue", confidence=0.95)


class FakeRetriever:
    def search(self, query: str, top_k: int = 5, exclude_hashes=None):
        return [RetrievedExample(score=0.9, message="I cannot log in", brand_reply="Please reset your password", conversation_id="1", brand_id="brand_a")]


class FakeReplyGenerator(GroundedReplyGenerator):
    def __init__(self):
        pass

    def generate(self, customer_message: str, intent: str, retrieved_examples):
        return ReplyResult(reply="Please reset your password.", grounding_examples=[example.__dict__ for example in retrieved_examples])

    def fallback_reply(self, retrieved_examples):
        return ReplyResult(reply="Please reset your password.", grounding_examples=[example.__dict__ for example in retrieved_examples])


def test_pipeline_output_structure():
    taxonomy = {"brand_id": "brand_a", "intents": [{"name": "login_issue", "description": "", "inclusion_criteria": "", "exclusion_criteria": ""}]}
    agent = SupportAgent(
        config=AppConfig(),
        taxonomy=taxonomy,
        retriever=FakeRetriever(),
        intent_model=FakeIntentModel(),
        reply_generator=FakeReplyGenerator(),
    )
    result = agent.run("I cannot log in")
    payload = result.__dict__
    assert set(payload.keys()) == {"intent", "intent_confidence", "retrieved_examples", "reply", "action", "reason", "llm_fallback_used", "llm_warning"}
    assert payload["intent"] == "login_issue"
    assert payload["llm_fallback_used"] is False
    assert payload["llm_warning"] is None
