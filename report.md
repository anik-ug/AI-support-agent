# Hiver SDE Intern Take-Home Report

## 1. Problem Framing

The task is to turn noisy customer-support threads into a small support agent for one brand. The agent must classify intent, retrieve similar historical cases, draft a grounded reply, and decide whether to auto-handle or escalate.

This workspace includes a 12-message synthetic Twitter-style smoke-test sample, so the complete pipeline can be exercised. It does not include enough real brand-specific data for credible evaluation; those steps remain explicitly pending rather than being invented.

## 2. What Good Means For This Brand

For a real selected brand, good means:

- Correctly identifying the customer’s support intent.
- Reusing historically consistent support behavior in the draft reply.
- Escalating when the issue is sensitive, ambiguous, or poorly supported by retrieved evidence.
- Avoiding invented refunds, timelines, account actions, or policy claims.

Because the available sample has only three conversations for the selected brand, there is no honest way to state brand-specific success criteria yet.

## 3. What We Chose Not To Build

I intentionally omitted production features that are irrelevant to the assignment:

- Frontend UI
- FastAPI service layer
- Authentication
- Real Twitter integration
- Ticketing integration
- Microservices
- Kubernetes or deployment tooling
- Streaming
- Production monitoring

The goal is a compact research prototype that can be explained live.

## 4. Dataset and Sampling

The code expects the Twitter export under `data/raw/` and discovers CSV, Parquet, Feather, JSON, and JSONL files recursively. The inspection script infers the text column, user fields, timestamps, and reply linkage columns instead of assuming a fixed schema.

A real run would:

- Inspect file names, columns, dtypes, duplicates, and missing values.
- Reconstruct conversations from reply links or thread IDs.
- Select one brand based on observed support volume and thread count.
- Sample a manageable reproducible subset with a fixed random seed.

On the smoke-test sample, `brand_a` is selected deterministically with three customer messages, three support messages, and three conversations. Those figures validate the mechanics only; statistics from the real export are still pending.

## 5. Intent Taxonomy

The repository includes a provisional six-intent taxonomy in `data/intent_taxonomy.json`, grounded in the smoke-test messages. It is explicitly not a final taxonomy and must be validated against actual customer messages from a selected real brand.

The intended process is:

- Inspect the brand’s real customer messages.
- Cluster similar issue types.
- Define approximately 6 to 8 intents with inclusion and exclusion rules.
- Write examples directly from observed messages.

## 6. System Architecture

The prototype is a simple pipeline:

1. Classify the incoming message into an intent.
2. Retrieve similar historical conversations with sentence embeddings and FAISS.
3. Draft a reply using retrieved historical evidence.
4. Apply escalation rules.

The core files are:

- `src/intent_classifier.py`
- `src/retriever.py`
- `src/reply_generator.py`
- `src/escalation.py`
- `src/pipeline.py`

## 7. Baselines

Two baselines are implemented:

- Majority-class baseline.
- TF-IDF + Logistic Regression baseline.

These are sufficient for comparing the final system against trivial and simple supervised approaches.

## 8. Evaluation Methodology

Intent evaluation uses accuracy, macro precision, macro recall, macro F1, per-class metrics, and confusion matrices.

Escalation evaluation uses precision, recall, F1, and confusion matrices.

Reply evaluation uses an LLM-as-judge rubric with five scored dimensions:

- Relevance
- Helpfulness
- Groundedness
- Brand consistency
- Safety

Human agreement is designed as a manual rating workflow for 30 to 50 replies, compared against judge ratings with Spearman correlation.

## 9. Results

No meaningful dataset-driven results are reported here. That is intentional and honest: the available smoke-test sample is far too small, so any quality metric claims would be fabricated.

The repository can produce real metrics once the full dataset is placed under `data/raw/`, the taxonomy is reviewed, and `data/golden_eval.csv` is manually annotated.

## 10. Top 5 Failure Modes

Failure-mode analysis is set up, but no real examples exist yet because there are no measured predictions. The likely categories for the real run are:

- Ambiguous messages
- Multiple issues in one tweet
- Noisy or typo-heavy messages
- Rare intents
- Poor retrieval or insufficient evidence

The report should be updated only after real evaluation outputs are available.

## 11. What Is Misleading About My Headline Number?

The most misleading thing about any eventual headline metric is that it will be based on a small hand-labeled eval set, not the full Twitter distribution. That means:

- Sampling bias may overstate performance on common issues.
- Old data may not reflect current support policy.
- LLM judge scores are not the same as customer satisfaction.
- Confidence is not calibrated.
- Retrieval quality can look better on held-out examples than on messy live traffic.

That limitation applies to the actual experimental setup, so the report must state it plainly once real numbers are added.

## 12. What I Would Do With One More Week

With one more week, I would:

- Label the golden evaluation set properly from the selected brand.
- Tune the intent taxonomy after inspecting more threads.
- Calibrate escalation thresholds.
- Run a small human-vs-judge agreement study.
- Improve failure analysis with concrete examples.
- Validate the retrieval and reply quality on more held-out cases.

## 13. Decision Log

- Decision: Keep the implementation as a Python research prototype.
  Why: It is the fastest honest path to the assignment goals.
  Trade-off: Less production polish, but more clarity and easier live explanation.

- Decision: Make dataset inspection schema-driven.
  Why: The Twitter export schema is not guaranteed.
  Trade-off: More defensive code, but fewer assumptions.

- Decision: Use conversation-level sampling and train/eval split.
  Why: Prevent leakage across messages from the same thread.
  Trade-off: Smaller effective sample, but more honest evaluation.

- Decision: Store the golden set as a manual template.
  Why: Human labels cannot be fabricated.
  Trade-off: Metrics remain pending until labeling is done.

- Decision: Use macro F1 as the primary classification metric.
  Why: The class distribution is likely imbalanced.
  Trade-off: Less emphasis on the dominant class.

- Decision: Include a majority baseline.
  Why: It gives a sanity-check floor.
  Trade-off: Very weak model, but useful comparator.

- Decision: Include TF-IDF + Logistic Regression as a simple baseline.
  Why: It is cheap, interpretable, and strong enough for short texts.
  Trade-off: Less semantic reasoning than embeddings or LLMs.

- Decision: Use sentence-transformers with FAISS for retrieval.
  Why: Lightweight and fast for support-message similarity.
  Trade-off: Requires embedding model dependency, but no heavy infrastructure.

- Decision: Use a strict JSON LLM interface.
  Why: It is safer to validate and easier to parse.
  Trade-off: More prompt discipline, but fewer malformed responses.

- Decision: Escalate on low confidence or weak evidence.
  Why: Support systems should avoid over-claiming on uncertain cases.
  Trade-off: More human handoff, but safer behavior.

- Decision: Provide fallback behavior when the LLM is unavailable.
  Why: The repo should still run without secrets.
  Trade-off: Output quality degrades, but the pipeline remains executable.

- Decision: Keep the report honest about missing measurements.
  Why: Fabricated metrics are worse than incomplete ones.
  Trade-off: The repo looks less polished now, but is defensible.

## References

- Customer Support on Twitter dataset: Kaggle `thoughtvector/customer-support-on-twitter`
- `sentence-transformers/all-MiniLM-L6-v2`
- FAISS
- scikit-learn
- OpenAI-compatible chat completions, if configured
