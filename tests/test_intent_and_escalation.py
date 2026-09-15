from __future__ import annotations

from dataclasses import dataclass

from src.escalation import decide_escalation
from src.intent_classifier import LLMIntentClassifier, MajorityIntentClassifier
from src.llm_client import LLMClient


class FakeLLMClient:
    available = True

    def __init__(self, response_text: str):
        self.response_text = response_text

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0):
        @dataclass
        class Response:
            text: str
            parsed: dict | None = None

        return Response(text=self.response_text, parsed=None)


def test_escalation_logic_flags_sensitive_requests():
    decision = decide_escalation(
        intent="login_issue",
        intent_confidence=0.9,
        top_similarity=0.9,
        retrieved_count=3,
        text="My password and OTP are not working",
        allowed_intents=["login_issue"],
        min_intent_confidence=0.5,
        min_similarity=0.4,
        min_retrieval_hits=2,
    )
    assert decision.action == "ESCALATE"


def test_llm_intent_classifier_falls_back_on_malformed_json():
    taxonomy = {"intents": [{"name": "login_issue", "description": "", "inclusion_criteria": "", "exclusion_criteria": ""}]}
    fallback = MajorityIntentClassifier().fit(["sample"], ["login_issue"])
    classifier = LLMIntentClassifier(taxonomy=taxonomy, llm_client=FakeLLMClient("not json"), fallback_model=fallback)
    prediction = classifier.predict_one("I cannot sign in")
    assert prediction.intent == "login_issue"
    assert prediction.used_fallback is True
    assert prediction.confidence == 0.5
