"""Unit tests for data loading, deduplication, and feature engineering."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import load_raw_data, clean_and_deduplicate, save_processed_data
from src.features import engineer_features, prepare_features_and_target, scale_features


EXPECTED_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]


def test_load_raw_data_exists():
    """Verify that raw dataset exists, loads properly, and is non-empty."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1599, f"Expected 1599 records, got {len(df)}"
    assert len(df.columns) == 12


def test_raw_data_expected_columns():
    """Verify all 12 UCI physicochemical columns exist."""
    df = load_raw_data()
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"Missing expected column: {col}"


def test_raw_data_no_missing_values():
    """Ensure raw dataset has no NaN or Null entries."""
    df = load_raw_data()
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected null values"


def test_clean_and_deduplicate():
    """Verify deduplication removes redundant observations."""
    df = load_raw_data()
    df_cleaned, dupes_removed = clean_and_deduplicate(df)
    assert dupes_removed == 240, f"Expected 240 duplicates, found {dupes_removed}"
    assert len(df_cleaned) == 1359
    assert df_cleaned.duplicated().sum() == 0


def test_save_and_reload_processed_data(tmp_path):
    """Verify saving processed dataframe preserves schema and data."""
    sample_df = pd.DataFrame({
        "alcohol": [10.5, 11.2],
        "quality": [5, 7],
    })
    target_file = tmp_path / "test_out.csv"
    save_processed_data(sample_df, str(target_file))

    assert target_file.exists()
    reloaded = pd.read_csv(target_file)
    assert reloaded.shape == (2, 2)
    assert list(reloaded.columns) == ["alcohol", "quality"]


def test_engineer_features():
    """Verify that engineered features are appended and logically valid."""
    df = load_raw_data().head(20)
    df_feat = engineer_features(df)

    assert "total_acidity" in df_feat.columns
    assert "bound_sulfur_dioxide" in df_feat.columns
    assert "free_sulfur_ratio" in df_feat.columns
    assert "alcohol_to_sugar_ratio" in df_feat.columns

    # Verify mathematical correctness
    sample = df_feat.iloc[0]
    expected_total_acidity = sample["fixed acidity"] + sample["volatile acidity"] + sample["citric acid"]
    assert np.isclose(sample["total_acidity"], expected_total_acidity)
    assert sample["bound_sulfur_dioxide"] >= 0
    assert 0 <= sample["free_sulfur_ratio"] <= 1.0


def test_prepare_features_and_target_binary():
    """Verify binary target transformation with quality threshold."""
    df = load_raw_data()
    X, y = prepare_features_and_target(df, binary=True, quality_threshold=7)

    assert "quality" not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y)


def test_load_raw_data_file_not_found():
    """Verify load_raw_data raises FileNotFoundError for non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("non_existent_file.csv")


def test_clean_and_deduplicate_reset_index():
    """Verify deduplication returns a continuous 0-based index."""
    df = load_raw_data()
    df_cleaned, _ = clean_and_deduplicate(df)
    assert list(df_cleaned.index) == list(range(len(df_cleaned)))


def test_prepare_features_and_target_missing_column():
    """Verify prepare_features_and_target raises KeyError when target column is missing."""
    df = load_raw_data().drop(columns=["quality"])
    with pytest.raises(KeyError):
        prepare_features_and_target(df, target_col="quality")


def test_scale_features_unknown_method():
    """Verify scale_features raises ValueError for unsupported scaling method."""
    df = load_raw_data().drop(columns=["quality"]).head(10)
    with pytest.raises(ValueError, match="Unknown scaling method"):
        scale_features(df, method="unsupported_method")
