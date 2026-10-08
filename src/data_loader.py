"""Data loading and preprocessing utilities for Red Wine Quality Analytics."""

import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd


DEFAULT_RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "winequality-red.csv"
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load the raw Red Wine Quality dataset.

    Args:
        filepath: Path to the raw CSV file. Defaults to data/raw/winequality-red.csv.

    Returns:
        pd.DataFrame: Raw wine quality dataset.

    Raises:
        FileNotFoundError: If the raw data file does not exist.
    """
    path = Path(filepath) if filepath else DEFAULT_RAW_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    # The UCI Red Wine Quality dataset is delimited by semicolons
    df = pd.read_csv(path, sep=";")
    # Strip any potential leading/trailing quotes or whitespace from column names
    df.columns = [col.strip().strip('"') for col in df.columns]
    return df


def clean_and_deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate records and handle missing values.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, count of removed duplicates).
    """
    initial_rows = len(df)
    # Deduplicate rows
    df_cleaned = df.drop_duplicates().copy()
    duplicates_removed = initial_rows - len(df_cleaned)

    # Validate missing values (UCI dataset is clean, but guard against nulls)
    if df_cleaned.isnull().any().any():
        df_cleaned = df_cleaned.dropna()

    return df_cleaned, duplicates_removed


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save a processed DataFrame to a CSV file.

    Args:
        df: Processed DataFrame.
        output_path: Destination file path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def run_pipeline(
    raw_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, int]:
    """Execute the data loading and deduplication pipeline.

    Args:
        raw_path: Path to the raw CSV file.
        output_dir: Directory where processed CSVs should be saved.

    Returns:
        Tuple of (cleaned DataFrame, count of removed duplicates).
    """
    df_raw = load_raw_data(raw_path)
    df_cleaned, dupes_removed = clean_and_deduplicate(df_raw)

    out_dir = Path(output_dir) if output_dir else DEFAULT_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = out_dir / "winequality-cleaned.csv"
    save_processed_data(df_cleaned, str(cleaned_path))

    return df_cleaned, dupes_removed


if __name__ == "__main__":
    print("Executing data loading pipeline...")
    cleaned, dupes = run_pipeline()
    print(f"Loaded raw dataset, removed {dupes} duplicate rows.")
    print(f"Cleaned dataset saved with shape: {cleaned.shape}")
