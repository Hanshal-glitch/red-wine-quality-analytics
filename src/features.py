"""Feature engineering, transformations, and scaling utilities."""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive physicochemical features from base wine attributes.

    Engineered features:
        - total_acidity: fixed + volatile + citric acid
        - bound_sulfur_dioxide: total sulfur dioxide - free sulfur dioxide
        - free_sulfur_ratio: free SO2 / (total SO2 + 1e-5)
        - alcohol_to_sugar_ratio: alcohol / (residual sugar + 1e-5)

    Args:
        df: Input DataFrame containing base wine attributes.

    Returns:
        pd.DataFrame: DataFrame with engineered features appended.
    """
    df_out = df.copy()

    # Total acidity combination
    df_out["total_acidity"] = (
        df_out["fixed acidity"] + df_out["volatile acidity"] + df_out["citric acid"]
    )

    # Bound SO2 (SO2 bound to aldehydes/sugars)
    df_out["bound_sulfur_dioxide"] = np.maximum(
        0.0, df_out["total sulfur dioxide"] - df_out["free sulfur dioxide"]
    )

    # Ratio of free SO2 to total SO2 (antioxidant effectiveness)
    df_out["free_sulfur_ratio"] = df_out["free sulfur dioxide"] / (
        df_out["total sulfur dioxide"] + 1e-5
    )

    # Balance between alcohol content and residual sugar
    df_out["alcohol_to_sugar_ratio"] = df_out["alcohol"] / (
        df_out["residual sugar"] + 1e-5
    )

    return df_out


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = "quality",
    binary: bool = True,
    quality_threshold: int = 7,
    with_engineered_features: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataset into features (X) and target (y).

    Args:
        df: Input DataFrame.
        target_col: Target column name. Default 'quality'.
        binary: If True, convert quality into binary (1 if >= threshold else 0).
        quality_threshold: Cutoff score for 'good' wine (default 7).
        with_engineered_features: If True, apply engineer_features first.

    Returns:
        Tuple of (X, y).
    """
    df_work = engineer_features(df) if with_engineered_features else df.copy()

    if target_col not in df_work.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    if binary:
        y = (df_work[target_col] >= quality_threshold).astype(int)
    else:
        y = df_work[target_col].copy()

    X = df_work.drop(columns=[target_col])
    return X, y


def scale_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    method: str = "standard",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], object]:
    """Fit scaler on X_train and transform both train and test.

    Args:
        X_train: Training features.
        X_test: Optional test features.
        method: Scaler type ('standard', 'robust', or 'minmax').

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method '{method}'. Choose 'standard', 'robust', or 'minmax'.")

    cols = X_train.columns
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=cols, index=X_train.index
    )

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=cols, index=X_test.index
        )

    return X_train_scaled, X_test_scaled, scaler


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    binary: bool = True,
    quality_threshold: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object]:
    """Complete split and scale pipeline.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler).
    """
    X, y = prepare_features_and_target(
        df, binary=binary, quality_threshold=quality_threshold
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
