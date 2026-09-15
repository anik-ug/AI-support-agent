from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def sample_twitter_df() -> pd.DataFrame:
    fixture = Path(__file__).parent / "fixtures" / "support_twitter_sample.csv"
    return pd.read_csv(fixture)
