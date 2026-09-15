from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from src.common import ensure_dir, write_json


@dataclass
class RetrievedExample:
    score: float
    message: str
    brand_reply: str
    conversation_id: str | None = None
    brand_id: str | None = None
    customer_message_hash: str | None = None
    brand_reply_hash: str | None = None


class SentenceTransformerRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.backend = "sentence_transformers"
        self.index = None
        self.embeddings: np.ndarray | None = None
        self.examples: list[dict[str, Any]] = []
        self.dimension: int | None = None

    @staticmethod
    def _use_faiss() -> bool:
        """FAISS is optional; NumPy cosine search is the portable default."""
        return os.getenv("USE_FAISS", "").strip().lower() in {"1", "true", "yes"}

    def _load_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer

                # A local model keeps the default demo deterministic and avoids a
                # network retry loop. Set up the model cache beforehand to use it.
                self.model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self.backend = "hashing"
        return self.model

    @staticmethod
    def _hash_encode(texts: list[str]) -> np.ndarray:
        """Local dependency-free semantic fallback for offline demo use."""
        vectorizer = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
        )
        return vectorizer.transform(texts).toarray().astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "hashing":
            return self._hash_encode(texts)
        model = self._load_model()
        if model is None:
            return self._hash_encode(texts)
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)

    def fit(self, examples: pd.DataFrame) -> "SentenceTransformerRetriever":
        messages = examples["customer_message"].astype(str).tolist()
        self.embeddings = self.encode(messages)
        self.dimension = int(self.embeddings.shape[1])
        if self._use_faiss():
            try:
                import faiss

                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(self.embeddings)
            except Exception:
                self.index = None
        self.examples = examples.to_dict(orient="records")
        return self

    def search(self, query: str, top_k: int = 5, exclude_hashes: set[str] | None = None) -> list[RetrievedExample]:
        if self.embeddings is None or not self.examples:
            return []
        query_embedding = self.encode([query])
        exclude_hashes = exclude_hashes or set()
        scores = self._search_scores(query_embedding, top_k=max(top_k * 5, top_k))
        results: list[RetrievedExample] = []
        for index, score in scores:
            example = self.examples[int(index)]
            if example.get("customer_message_hash") in exclude_hashes:
                continue
            results.append(
                RetrievedExample(
                    score=float(score),
                    message=str(example.get("customer_message", "")),
                    brand_reply=str(example.get("brand_reply", "")),
                    conversation_id=example.get("conversation_id"),
                    brand_id=example.get("brand_id"),
                    customer_message_hash=example.get("customer_message_hash"),
                    brand_reply_hash=example.get("brand_reply_hash"),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def _search_scores(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.index is not None:
            scores, indices = self.index.search(query_embedding, top_k)
            return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
        assert self.embeddings is not None
        query = query_embedding[0]
        scores = self.embeddings @ query
        indices = np.argsort(-scores)[:top_k]
        return [(int(idx), float(scores[idx])) for idx in indices]

    def save(self, output_dir: Path) -> None:
        ensure_dir(output_dir)
        metadata = {
            "model_name": self.model_name,
            "backend": self.backend,
            "dimension": self.dimension,
            "examples": self.examples,
        }
        write_json(output_dir / "retriever_metadata.json", metadata)
        if self.embeddings is not None:
            np.save(output_dir / "embeddings.npy", self.embeddings)
        if self.index is not None:
            import faiss

            faiss.write_index(self.index, str(output_dir / "faiss.index"))

    @classmethod
    def load(cls, output_dir: Path) -> "SentenceTransformerRetriever":
        retriever = cls()
        metadata_path = output_dir / "retriever_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing retriever metadata at {metadata_path}")
        import json

        metadata = json.loads(metadata_path.read_text())
        retriever.model_name = metadata["model_name"]
        retriever.backend = metadata.get("backend", "sentence_transformers")
        retriever.dimension = metadata["dimension"]
        retriever.examples = metadata["examples"]
        embeddings_path = output_dir / "embeddings.npy"
        if embeddings_path.exists():
            retriever.embeddings = np.load(embeddings_path)
        index_path = output_dir / "faiss.index"
        if index_path.exists() and retriever._use_faiss():
            try:
                import faiss

                retriever.index = faiss.read_index(str(index_path))
            except Exception:
                # Numpy cosine search remains available without the optional FAISS package.
                retriever.index = None
        return retriever
