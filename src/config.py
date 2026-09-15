from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.common import ensure_dir

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVALUATION_DIR = ROOT_DIR / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"
GOLDEN_EVAL_PATH = DATA_DIR / "golden_eval.csv"
INTENT_TAXONOMY_PATH = DATA_DIR / "intent_taxonomy.json"

@dataclass(frozen=True)
class EscalationThresholds:
    min_intent_confidence: float = 0.55
    min_similarity: float = 0.45
    min_retrieval_hits: int = 2

@dataclass(frozen=True)
class AppConfig:
    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    evaluation_dir: Path = EVALUATION_DIR
    results_dir: Path = RESULTS_DIR
    golden_eval_path: Path = GOLDEN_EVAL_PATH
    intent_taxonomy_path: Path = INTENT_TAXONOMY_PATH
    dataset_path: Path | None = None
    brand_id: str | None = None
    random_seed: int = 13
    sample_size: int = 2500
    max_conversations_per_brand: int = 1000
    golden_eval_size: int = 200
    eval_ratio: float = 0.2
    top_k_retrieval: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    llm_api_key: str | None = None
    use_faiss: bool = False
    escalation: EscalationThresholds = EscalationThresholds()

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()
        dataset_raw = os.getenv("DATASET_PATH", "").strip()
        dataset_path = Path(dataset_raw).expanduser() if dataset_raw else None
        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else (os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY"))
        return cls(
            dataset_path=dataset_path,
            brand_id=os.getenv("BRAND_ID", "").strip() or None,
            random_seed=int(os.getenv("RANDOM_SEED", "13")),
            sample_size=int(os.getenv("SAMPLE_SIZE", "2500")),
            max_conversations_per_brand=int(os.getenv("MAX_CONVERSATIONS_PER_BRAND", "1000")),
            golden_eval_size=int(os.getenv("GOLDEN_EVAL_SIZE", "200")),
            eval_ratio=float(os.getenv("EVAL_RATIO", "0.2")),
            top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "5")),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            llm_provider=provider,
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            openai_api_key=api_key if provider == "openai" else None,
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            use_faiss=os.getenv("USE_FAISS", "0").strip().lower() in {"1", "true", "yes"},
            escalation=EscalationThresholds(
                min_intent_confidence=float(os.getenv("MIN_INTENT_CONFIDENCE", "0.55")),
                min_similarity=float(os.getenv("MIN_SIMILARITY", "0.45")),
                min_retrieval_hits=int(os.getenv("MIN_RETRIEVAL_HITS", "2")),
            ),
        )

    def resolved_dataset_path(self) -> Path:
        if self.dataset_path and self.dataset_path.exists():
            return self.dataset_path
        candidates = sorted(self.raw_data_dir.rglob("*.csv")) if self.raw_data_dir.exists() else []
        candidates = [p for p in candidates if p.name != "support_twitter_sample.csv"] or candidates
        if candidates:
            return candidates[0]
        raise FileNotFoundError(
            "Dataset not found. Set DATASET_PATH in .env to the extracted twcs.csv path."
        )

def ensure_workspace_dirs() -> None:
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
        ensure_dir(path)
