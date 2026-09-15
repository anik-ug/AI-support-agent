from __future__ import annotations
import json
import re
import pandas as pd
from _bootstrap import add_project_root_to_path
add_project_root_to_path()
from src.config import AppConfig, ensure_workspace_dirs
from src.common import safe_json_loads, write_json
from src.llm_client import LLMClient

FALLBACK_INTENTS = [
    ("account_access", "Problems signing in, accessing, or managing an account."),
    ("billing_payment", "Questions or problems involving charges, payments, or invoices."),
    ("refund_return", "Requests or questions about refunds, returns, or money being returned."),
    ("order_delivery", "Questions or problems involving orders, shipping, delivery, or missing packages."),
    ("subscription_plan", "Questions or problems involving subscriptions, plans, renewals, or cancellation."),
    ("technical_issue", "Product, website, app, login-flow, or technical malfunction reports."),
    ("general_support", "General product or support questions that do not fit another intent."),
    ("escalation_followup", "Repeated contact, unresolved issues, complaints, or requests for escalation."),
]

def fallback_taxonomy(brand_id: str, schema: dict) -> dict:
    intents=[]
    for name, desc in FALLBACK_INTENTS:
        intents.append({"name": name, "description": desc, "inclusion_criteria": [desc], "exclusion_criteria": [], "examples": []})
    return {"status":"active", "brand_id":brand_id, "schema":schema, "source":"deterministic fallback; human review required", "intents":intents}

def main():
    config=AppConfig.from_env(); ensure_workspace_dirs()
    pairs=pd.read_csv(config.processed_data_dir/"train_pairs.csv")
    selection=json.loads((config.processed_data_dir/"brand_selection.json").read_text())
    schema=json.loads((config.processed_data_dir/"schema.json").read_text())
    messages=pairs.get("customer_message", pd.Series(dtype=str)).dropna().astype(str).drop_duplicates()
    sample=messages.sample(min(80,len(messages)), random_state=config.random_seed).tolist() if len(messages) else []
    taxonomy=None
    client=LLMClient(config.llm_provider, config.llm_model, config.openai_api_key if config.llm_provider=="openai" else (config.gemini_api_key or config.llm_api_key))
    if client.available and sample:
        system="You design a compact customer-support intent taxonomy from real messages. Use 6-10 mutually understandable intents. Do not invent brand policies. Return JSON only."
        user=("Create a taxonomy for these customer messages. Each intent must have name, description, inclusion_criteria, exclusion_criteria, examples. "
              "Names should be short snake_case labels. Include a general_support or other intent if needed.\n\n"+"\n".join(f"- {m}" for m in sample))
        try:
            response=client.chat(system,user,temperature=0.0)
            payload=response.parsed or safe_json_loads(response.text)
            intents=payload.get("intents", payload if isinstance(payload,list) else [])
            if 5 <= len(intents) <= 12:
                taxonomy={"status":"active", "brand_id":str(selection["brand_value"]), "schema":schema, "source":"LLM-generated candidate taxonomy; human review required", "intents":intents}
        except Exception as exc:
            print(f"Taxonomy LLM generation failed: {exc}. Using deterministic fallback.")
    if taxonomy is None:
        taxonomy=fallback_taxonomy(str(selection["brand_value"]), schema)
    write_json(config.intent_taxonomy_path, taxonomy)
    print(f"Taxonomy written: {config.intent_taxonomy_path}")
    print(f"Brand: {selection['brand_value']}")
    print(f"Intents: {len(taxonomy['intents'])}")
    print("IMPORTANT: review the taxonomy before treating it as final evaluation ground truth.")

if __name__ == "__main__": main()
