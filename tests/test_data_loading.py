from __future__ import annotations

from src.config import AppConfig
from src.data_loader import infer_schema, load_raw_dataset
from src.data_processing import assign_conversation_ids, clean_raw_frame, derive_brand_column, derive_message_role, prepare_brand_dataset


def test_load_raw_dataset_handles_fixture(sample_twitter_df):
    schema = infer_schema(sample_twitter_df)
    assert schema.text_col == "text"
    assert schema.user_col == "author_id"
    assert schema.inbound_col == "inbound"
    assert schema.tweet_id_col == "tweet_id"
    assert schema.reply_to_col == "response_tweet_id"


def test_preprocessing_reconstructs_conversations(sample_twitter_df):
    config = AppConfig()
    schema = infer_schema(sample_twitter_df)
    cleaned = clean_raw_frame(sample_twitter_df, schema)
    cleaned["conversation_id"] = assign_conversation_ids(cleaned)
    cleaned["role"] = derive_message_role(cleaned)
    cleaned["brand_id"] = derive_brand_column(cleaned)
    assert cleaned["conversation_id"].nunique() < len(cleaned)
    assert set(cleaned["role"].unique()) == {"customer", "brand"}
    assert cleaned["brand_id"].nunique() >= 2


def test_prepare_brand_dataset_selects_brand(sample_twitter_df):
    config = AppConfig(sample_size=10)
    brand_df, brand_selection, schema = prepare_brand_dataset(config, raw_df=sample_twitter_df)
    assert not brand_df.empty
    assert brand_selection.brand_value in set(brand_df["brand_id"].astype(str))
    assert schema.text_col == "text"
