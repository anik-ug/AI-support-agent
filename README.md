# Hiver Support Agent — Evaluation-First Customer Support AI

An evaluation-first customer-support agent for the Hiver SDE Intern take-home. It uses the **Customer Support on Twitter (TWCS)** dataset, selects one support brand, reconstructs support conversations, builds a compact intent taxonomy, retrieves historical resolutions, drafts grounded replies, and decides **AUTO_HANDLE vs ESCALATE** with an explicit reason.

## What you get

- Real TWCS CSV support (including the 493 MB Kaggle CSV).
- Automatic inspection and support-account ranking.
- Automatic brand selection, or pin one with `BRAND_ID`.
- Conversation reconstruction using reply relationships.
- Customer→brand training/evaluation pairs.
- Candidate intent taxonomy generated from the selected brand's messages (LLM when configured; deterministic fallback otherwise).
- 200-row golden evaluation template.
- Majority + TF-IDF/Logistic Regression baselines.
- Sentence-Transformers retrieval with NumPy fallback; optional FAISS.
- Grounded LLM reply generation.
- Rule-based safety/escalation gate with reasons.
- LLM judge + human agreement utilities.
- Tests and reproducible artifacts.

## 1. Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 2. `.env` — the only configuration you need

For the Kaggle file you extracted to `~/Downloads/twcs/twcs.csv`:

```dotenv
DATASET_PATH=/Users/anikkumar/Downloads/twcs/twcs.csv
BRAND_ID=
RANDOM_SEED=13
SAMPLE_SIZE=2500
MAX_CONVERSATIONS_PER_BRAND=1000
GOLDEN_EVAL_SIZE=200
EVAL_RATIO=0.20
TOP_K_RETRIEVAL=5
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
USE_FAISS=0
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=
LLM_API_KEY=
MIN_INTENT_CONFIDENCE=0.55
MIN_SIMILARITY=0.45
MIN_RETRIEVAL_HITS=2
```

If you use Gemini instead:

```dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=
```

Do **not** commit `.env` or the dataset.

## 3. One-command data preparation

```bash
python scripts/run_all.py
```

This runs:

1. `inspect_dataset.py` — ranks outbound support authors.
2. `prepare_data.py` — selects a brand, reconstructs conversations, creates customer→brand pairs, and writes a 200-example golden template.
3. `generate_taxonomy.py` — creates a candidate taxonomy from real brand messages.
4. `build_index.py` — builds the retrieval artifact.

Generated files:

```text
data/processed/brand_conversations.csv
data/processed/brand_selection.json
data/processed/schema.json
data/processed/train_pairs.csv
data/processed/eval_pairs.csv
data/processed/retriever/
data/intent_taxonomy.json
data/golden_eval.csv
```

### Important evaluation step

The assignment explicitly requires a **150–250 example hand-labelled golden set**. The script creates the annotation sheet, but a program cannot honestly replace human labeling. Review `data/intent_taxonomy.json`, then manually fill these columns in `data/golden_eval.csv`:

- `intent`
- `expected_action` (`AUTO_HANDLE` or `ESCALATE`)
- `difficulty`
- `notes`

Keep the golden examples isolated from retrieval/training data.

## 4. Run the agent

```bash
python scripts/run_demo.py "I was charged twice for my order"
```

The JSON output contains:

- predicted intent + confidence
- retrieved historical examples
- drafted reply
- `AUTO_HANDLE` or `ESCALATE`
- escalation reason
- whether an LLM fallback was used

## 5. Evaluation

Run tests:

```bash
PYTHONPATH=. pytest -q
```

Once the golden set is hand-labelled, use the evaluation modules under `evaluation/`. Do not report metrics until the golden set is complete.

## Architecture

```text
Customer message
      ↓
Intent classification
      ↓
Historical-resolution retrieval
      ↓
Safety / confidence gate
      ↓
AUTO_HANDLE ──→ Grounded reply
      │
      └────────→ ESCALATE + reason
```

## Design decisions

- **Evaluation isolation:** golden examples are not used as retrieval evidence.
- **Conversation-level splitting:** avoids putting messages from the same thread in both train and eval.
- **Grounding:** the reply prompt is constrained by retrieved historical support behavior.
- **Safety gate:** low confidence, weak retrieval, sensitive-account requests, unsupported intents, and multi-issue messages can be escalated.
- **Fallback:** if the LLM is unavailable, the system remains runnable instead of pretending a live LLM was used.

## What is not automated on purpose

The take-home asks for human-labelled evaluation and human-vs-LLM-judge agreement. Those are evidence-generation steps, not things the system should fabricate. The repository therefore automates **candidate generation and measurement**, while requiring human review for the final golden labels and taxonomy.

## Dataset

Customer Support on Twitter (TWCS), Kaggle: `thoughtvector/customer-support-on-twitter`.
