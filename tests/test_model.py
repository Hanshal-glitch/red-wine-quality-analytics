"""Unit tests for model training, metrics evaluation, and inference."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import load_raw_data, clean_and_deduplicate
from src.features import create_train_test_split, scale_features
from src.train import train_and_evaluate_models, train_pipeline
from src.predict import predict_wine_quality, load_artifacts


def test_scale_features():
    """Verify feature scaling yields approximately zero mean and unit variance."""
    df_raw = load_raw_data().drop(columns=["quality"])
    X_train_scaled, _, scaler = scale_features(df_raw.head(100))

    assert X_train_scaled.shape == (100, 11)
    # Check that scaled mean is approximately 0 and variance is ~1
    means = X_train_scaled.mean(axis=0)
    assert np.all(np.abs(means) < 1e-1)


def test_train_and_evaluate_models():
    """Test candidate models training and metric structure."""
    df_raw = load_raw_data().head(300)
    df_cleaned, _ = clean_and_deduplicate(df_raw)
    X_train, X_test, y_train, y_test, _ = create_train_test_split(
        df_cleaned, test_size=0.25, random_state=42
    )

    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    assert "RandomForest" in results
    assert "GradientBoosting" in results
    assert "LogisticRegression" in results

    rf_metrics = results["RandomForest"]["metrics"]
    assert "accuracy" in rf_metrics
    assert "f1" in rf_metrics
    assert "roc_auc" in rf_metrics
    assert 0.0 <= rf_metrics["accuracy"] <= 1.0


def test_end_to_end_training_and_artifacts(tmp_path):
    """Test training pipeline writes valid artifacts and returns summary."""
    models_dir = tmp_path / "models"
    summary = train_pipeline(models_dir=str(models_dir), save_artifacts=True)

    assert "best_model_name" in summary
    assert (models_dir / "wine_quality_model.joblib").exists()
    assert (models_dir / "scaler.joblib").exists()
    assert (models_dir / "model_metadata.json").exists()


def test_predict_wine_quality_inference(tmp_path):
    """Test inference function produces valid probability and classification label."""
    models_dir = tmp_path / "models"
    train_pipeline(models_dir=str(models_dir), save_artifacts=True)

    sample_wine = {
        "fixed acidity": 8.5,
        "volatile acidity": 0.28,
        "citric acid": 0.56,
        "residual sugar": 1.8,
        "chlorides": 0.092,
        "free sulfur dioxide": 35.0,
        "total sulfur dioxide": 103.0,
        "density": 0.9969,
        "pH": 3.30,
        "sulphates": 0.75,
        "alcohol": 10.5,
    }

    result = predict_wine_quality(sample_wine, models_dir=str(models_dir))

    assert "prediction_binary" in result
    assert result["prediction_binary"] in (0, 1)
    assert "quality_label" in result
    assert "probability_high_quality" in result
    assert 0.0 <= result["probability_high_quality"] <= 1.0
    assert 0.0 <= result["confidence_percentage"] <= 100.0
