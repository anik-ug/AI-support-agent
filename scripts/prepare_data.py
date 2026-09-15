from __future__ import annotations
from _bootstrap import add_project_root_to_path
add_project_root_to_path()
from src.config import AppConfig, ensure_workspace_dirs
from src.data_processing import build_customer_brand_pairs, build_golden_eval_template, prepare_brand_dataset, save_processed_artifacts, split_train_eval

def main():
    config = AppConfig.from_env(); ensure_workspace_dirs()
    brand_df, selection, schema = prepare_brand_dataset(config)
    save_processed_artifacts(config, brand_df, selection, schema)
    pairs = build_customer_brand_pairs(brand_df)
    if config.max_conversations_per_brand > 0 and len(pairs) > config.max_conversations_per_brand:
        pairs = pairs.sample(config.max_conversations_per_brand, random_state=config.random_seed).reset_index(drop=True)
    train_df, eval_df = split_train_eval(pairs, config.eval_ratio, config.random_seed)
    train_df.to_csv(config.processed_data_dir / "train_pairs.csv", index=False)
    eval_df.to_csv(config.processed_data_dir / "eval_pairs.csv", index=False)
    golden = build_golden_eval_template(eval_df if not eval_df.empty else pairs, config.golden_eval_path, sample_size=config.golden_eval_size, seed=config.random_seed)
    print(f"Dataset: {config.resolved_dataset_path()}")
    print(f"Selected brand/support author: {selection.brand_value}")
    print(f"Support messages: {selection.support_message_count:,}")
    print(f"Customer messages: {selection.customer_message_count:,}")
    print(f"Conversations: {selection.conversation_count:,}")
    print(f"Usable customer→brand pairs: {len(pairs):,}")
    print(f"Train pairs: {len(train_df):,}; Eval pairs: {len(eval_df):,}")
    print(f"Golden template: {config.golden_eval_path} ({len(golden)} rows)")

if __name__ == "__main__": main()
