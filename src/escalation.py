from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class EscalationDecision:
    action: str
    reason: str


def contains_account_sensitive_request(text: str) -> bool:
    lowered = text.lower()
    sensitive_terms = ["password", "otp", "two-factor", "2fa", "security code", "ssn", "credit card", "bank account", "private message", "dm us"]
    return any(term in lowered for term in sensitive_terms)


def has_multiple_issues(text: str) -> bool:
    lowered = text.lower()
    separators = [" and ", " also ", " but ", " yet ", " plus ", ";"]
    return sum(term in lowered for term in separators) >= 2


def decide_escalation(
    *,
    intent: str,
    intent_confidence: float,
    top_similarity: float,
    retrieved_count: int,
    text: str,
    allowed_intents: Iterable[str],
    min_intent_confidence: float,
    min_similarity: float,
    min_retrieval_hits: int,
) -> EscalationDecision:
    if intent not in set(allowed_intents):
        return EscalationDecision("ESCALATE", "Intent is unsupported or unknown for this brand.")
    if intent_confidence < min_intent_confidence:
        return EscalationDecision("ESCALATE", f"Intent confidence {intent_confidence:.2f} is below the configured threshold {min_intent_confidence:.2f}.")
    if retrieved_count < min_retrieval_hits:
        return EscalationDecision("ESCALATE", "Not enough historical examples were retrieved to support a safe auto-response.")
    if top_similarity < min_similarity:
        return EscalationDecision("ESCALATE", f"Top retrieval similarity {top_similarity:.2f} is below the configured threshold {min_similarity:.2f}.")
    if contains_account_sensitive_request(text):
        return EscalationDecision("ESCALATE", "The message appears to involve account-sensitive or private information.")
    if has_multiple_issues(text):
        return EscalationDecision("ESCALATE", "The message appears to contain multiple issues that should be reviewed by a human.")
    return EscalationDecision("AUTO_HANDLE", "Intent confidence and historical evidence are sufficient for an automatic reply.")
