from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.common import safe_json_loads
from src.llm_client import LLMClient


@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    raw_response: str | None = None
    used_fallback: bool = False


class MajorityIntentClassifier:
    def fit(self, texts: list[str], labels: list[str]) -> "MajorityIntentClassifier":
        counts = pd.Series(labels).value_counts()
        self.majority_intent_ = str(counts.index[0])
        self.classes_ = counts.index.astype(str).tolist()
        return self

    def predict_one(self, text: str) -> IntentPrediction:
        return IntentPrediction(intent=self.majority_intent_, confidence=1.0, used_fallback=True)

    def predict(self, texts: list[str]) -> list[IntentPrediction]:
        return [self.predict_one(text) for text in texts]


class TfidfIntentClassifier:
    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=13)),
            ]
        )

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfIntentClassifier":
        self.pipeline.fit(texts, labels)
        return self

    def predict_one(self, text: str) -> IntentPrediction:
        predicted = self.pipeline.predict([text])[0]
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.named_steps["clf"].classes_
        confidence = float(proba[np.where(classes == predicted)[0][0]])
        return IntentPrediction(intent=str(predicted), confidence=confidence)

    def predict(self, texts: list[str]) -> list[IntentPrediction]:
        return [self.predict_one(text) for text in texts]


def build_allowed_intents(taxonomy: dict[str, Any]) -> list[str]:
    if "intents" in taxonomy:
        return [str(intent["name"]) for intent in taxonomy["intents"]]
    return [str(key) for key in taxonomy.keys()]


def format_intent_prompt(taxonomy: dict[str, Any], customer_message: str) -> tuple[str, str]:
    intents = taxonomy.get("intents", [])
    intent_lines = []
    for intent in intents:
        intent_lines.append(
            f"- {intent['name']}: {intent.get('description', '')}\n  include: {intent.get('inclusion_criteria', '')}\n  exclude: {intent.get('exclusion_criteria', '')}"
        )
    system_prompt = (
        "You classify customer-support messages into one of the allowed intents. "
        "Return only strict JSON with keys intent and confidence. "
        "Confidence must be a 0.0 to 1.0 self-reported confidence, not a calibrated probability."
    )
    user_prompt = f"Allowed intents:\n{chr(10).join(intent_lines)}\n\nCustomer message:\n{customer_message}\n\nReturn JSON only."
    return system_prompt, user_prompt


class LLMIntentClassifier:
    def __init__(self, taxonomy: dict[str, Any], llm_client: LLMClient, fallback_model: MajorityIntentClassifier | TfidfIntentClassifier | None = None):
        self.taxonomy = taxonomy
        self.allowed_intents = set(build_allowed_intents(taxonomy))
        self.llm_client = llm_client
        self.fallback_model = fallback_model

    def _validate(self, payload: dict[str, Any]) -> IntentPrediction:
        intent = str(payload.get("intent", "UNKNOWN"))
        confidence = float(payload.get("confidence", 0.0))
        if intent not in self.allowed_intents:
            raise ValueError(f"Invalid intent returned by model: {intent}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        return IntentPrediction(intent=intent, confidence=confidence, raw_response=None)

    def _fallback_prediction(self, message: str) -> IntentPrediction:
        if self.fallback_model is None:
            raise RuntimeError("No fallback intent model is configured.")
        fallback = self.fallback_model.predict_one(message)
        fallback.used_fallback = True
        fallback.confidence = min(float(fallback.confidence), 0.5)
        return fallback

    def predict_one(self, message: str) -> IntentPrediction:
        system_prompt, user_prompt = format_intent_prompt(self.taxonomy, message)
        try:
            response = self.llm_client.chat(system_prompt, user_prompt, temperature=0.0)
            payload = response.parsed or safe_json_loads(response.text)
            return self._validate(payload)
        except Exception:
            if self.fallback_model is not None:
                return self._fallback_prediction(message)
            raise

    def predict(self, messages: list[str]) -> list[IntentPrediction]:
        return [self.predict_one(message) for message in messages]
