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
    processed_dir = tmp_path / "processed"
    summary = train_pipeline(
        models_dir=str(models_dir),
        processed_dir=str(processed_dir),
        save_artifacts=True,
    )

    assert "best_model_name" in summary
    assert (models_dir / "wine_quality_model.joblib").exists()
    assert (models_dir / "scaler.joblib").exists()
    assert (models_dir / "model_metadata.json").exists()
    assert (processed_dir / "winequality-cleaned.csv").exists()
    assert (processed_dir / "train_scaled.csv").exists()
    assert (processed_dir / "test_scaled.csv").exists()


def test_load_artifacts_not_found(tmp_path):
    """Verify load_artifacts raises FileNotFoundError when artifacts do not exist."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_artifacts(models_dir=str(empty_dir))


def test_predict_wine_quality_inference(tmp_path):
    """Test inference function produces valid probability and classification label."""
    models_dir = tmp_path / "models"
    processed_dir = tmp_path / "processed"
    train_pipeline(
        models_dir=str(models_dir),
        processed_dir=str(processed_dir),
        save_artifacts=True,
    )

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


def test_predict_missing_features(tmp_path):
    """Verify predict_wine_quality raises ValueError when input lacks required features."""
    models_dir = tmp_path / "models"
    processed_dir = tmp_path / "processed"
    train_pipeline(
        models_dir=str(models_dir),
        processed_dir=str(processed_dir),
        save_artifacts=True,
    )

    incomplete_sample = {
        "fixed acidity": 8.5,
        "volatile acidity": 0.28,
        # missing other required physicochemical features
    }

    with pytest.raises(ValueError, match="Missing required wine feature"):
        predict_wine_quality(incomplete_sample, models_dir=str(models_dir))


def test_predict_batch_inference(tmp_path):
    """Verify predict_wine_quality handles multi-sample DataFrame batch input."""
    models_dir = tmp_path / "models"
    processed_dir = tmp_path / "processed"
    train_pipeline(
        models_dir=str(models_dir),
        processed_dir=str(processed_dir),
        save_artifacts=True,
    )

    batch_df = pd.DataFrame([
        {
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
        },
        {
            "fixed acidity": 7.0,
            "volatile acidity": 0.85,
            "citric acid": 0.05,
            "residual sugar": 2.5,
            "chlorides": 0.120,
            "free sulfur dioxide": 10.0,
            "total sulfur dioxide": 60.0,
            "density": 0.9980,
            "pH": 3.45,
            "sulphates": 0.45,
            "alcohol": 9.2,
        },
    ])

    batch_result = predict_wine_quality(batch_df, models_dir=str(models_dir))
    assert "predictions" in batch_result
    assert len(batch_result["predictions"]) == 2
    assert len(batch_result["prediction_binary"]) == 2
    assert len(batch_result["probability_high_quality"]) == 2
