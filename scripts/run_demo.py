from __future__ import annotations

import argparse
import json

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from src.common import read_json
from src.config import AppConfig, ensure_workspace_dirs
from src.intent_classifier import MajorityIntentClassifier, LLMIntentClassifier, TfidfIntentClassifier
from src.llm_client import LLMClient
from src.pipeline import SupportAgent
from src.reply_generator import GroundedReplyGenerator
from src.retriever import SentenceTransformerRetriever


def load_taxonomy(config: AppConfig) -> dict:
    taxonomy = read_json(config.intent_taxonomy_path, default={})
    if not taxonomy or taxonomy.get("status") == "pending":
        raise FileNotFoundError(
            f"Intent taxonomy is not ready yet. Generate it from the brand sample and place it at {config.intent_taxonomy_path}."
        )
    return taxonomy


def load_intent_model(taxonomy: dict, config: AppConfig):
    llm_client = LLMClient(provider=config.llm_provider, model=config.llm_model, api_key=(
    config.openai_api_key
    if config.llm_provider == "openai"
    else (config.gemini_api_key or config.llm_api_key)
))
    train_path = config.processed_data_dir / "train_pairs.csv"
    fallback = None
    if train_path.exists():
        train_df = __import__("pandas").read_csv(train_path)
        if {"customer_message", "intent"}.issubset(train_df.columns) and train_df["intent"].notna().all() and train_df["intent"].nunique() >= 2:
            fallback = TfidfIntentClassifier().fit(train_df["customer_message"].astype(str).tolist(), train_df["intent"].astype(str).tolist())
    if fallback is None and taxonomy.get("intents"):
        fallback = MajorityIntentClassifier().fit(["sample"], [taxonomy["intents"][0]["name"]])
    return LLMIntentClassifier(taxonomy=taxonomy, llm_client=llm_client, fallback_model=fallback)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hiver support agent on a single customer message.")
    parser.add_argument("message", nargs="?", help="Customer message text")
    args = parser.parse_args()
    message = args.message or input("Customer message: ").strip()

    config = AppConfig.from_env()
    ensure_workspace_dirs()
    taxonomy = load_taxonomy(config)
    retriever = SentenceTransformerRetriever.load(config.processed_data_dir / "retriever")
    llm_client = LLMClient(provider=config.llm_provider, model=config.llm_model, api_key=(
    config.openai_api_key
    if config.llm_provider == "openai"
    else (config.gemini_api_key or config.llm_api_key)
))
    intent_model = load_intent_model(taxonomy, config)
    brand_name = str(taxonomy.get("brand_id", "selected_brand"))
    reply_generator = GroundedReplyGenerator(llm_client=llm_client, brand_name=brand_name)
    agent = SupportAgent(config=config, taxonomy=taxonomy, retriever=retriever, intent_model=intent_model, reply_generator=reply_generator)
    result = agent.run(message)
    if result.llm_warning:
        print(f"WARNING: {result.llm_warning}")
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
