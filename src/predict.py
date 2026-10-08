"""Model inference and prediction interface for wine quality."""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import json
import joblib
import pandas as pd

from src.features import engineer_features


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_artifacts(models_dir: Optional[str] = None):
    """Load serialized model, scaler, and metadata.

    Args:
        models_dir: Directory containing serialized artifacts.

    Returns:
        Tuple of (model, scaler, metadata).
    """
    path = Path(models_dir) if models_dir else MODELS_DIR
    model_path = path / "wine_quality_model.joblib"
    scaler_path = path / "scaler.joblib"
    meta_path = path / "model_metadata.json"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {path}. Please run src/train.py first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return model, scaler, metadata


def predict_wine_quality(
    wine_sample: Union[Dict[str, float], pd.DataFrame],
    models_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Predict wine quality category and probability for input physicochemical sample(s).

    Args:
        wine_sample: Dictionary of raw physicochemical metrics or DataFrame.
        models_dir: Directory containing model artifacts.

    Returns:
        Dictionary containing prediction label, score, and probability.
    """
    model, scaler, metadata = load_artifacts(models_dir)

    if isinstance(wine_sample, dict):
        df = pd.DataFrame([wine_sample])
    else:
        df = wine_sample.copy()

    # Apply feature engineering
    df_engineered = engineer_features(df)

    feature_cols = metadata.get("feature_names", list(df_engineered.columns))
    # Keep only expected features in the right order
    df_aligned = df_engineered[feature_cols]

    # Scale features
    scaled_values = scaler.transform(df_aligned)

    predictions = model.predict(scaled_values)
    probabilities = (
        model.predict_proba(scaled_values)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    pred_class = int(predictions[0])
    prob_val = float(probabilities[0]) if probabilities is not None else 0.0

    return {
        "prediction_binary": pred_class,
        "quality_label": "High Quality (Score >= 7)" if pred_class == 1 else "Normal / Low Quality (Score < 7)",
        "probability_high_quality": prob_val,
        "confidence_percentage": round(prob_val * 100, 2),
    }


if __name__ == "__main__":
    sample = {
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
    print("Testing sample prediction:")
    try:
        res = predict_wine_quality(sample)
        print("Prediction result:", res)
    except FileNotFoundError as e:
        print(f"Notice: {e}")
