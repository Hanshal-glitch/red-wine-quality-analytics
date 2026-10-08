"""Model training, evaluation, and artifact serialization."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from src.data_loader import load_raw_data, clean_and_deduplicate, save_processed_data
from src.features import create_train_test_split


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def train_and_evaluate_models(
    X_train, X_test, y_train, y_test
) -> Dict[str, Dict[str, Any]]:
    """Train multiple candidate classifiers and evaluate test performance.

    Args:
        X_train: Scaled training features.
        X_test: Scaled testing features.
        y_train: Training labels.
        y_test: Testing labels.

    Returns:
        Dictionary mapping model names to fitted model and performance metrics.
    """
    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=42, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
    }

    results = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Determine predict_proba if available
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_proba))
        else:
            auc = 0.0

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": auc,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

        results[name] = {
            "model": model,
            "metrics": metrics,
        }

    return results


def train_pipeline(
    raw_path: Optional[str] = None,
    models_dir: Optional[str] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """End-to-end training pipeline.

    Args:
        raw_path: Optional custom path to raw dataset.
        models_dir: Optional custom directory to save trained model artifacts.
        save_artifacts: Whether to save artifacts to disk.

    Returns:
        Summary dictionary with metrics and best model name.
    """
    out_models = Path(models_dir) if models_dir else MODELS_DIR
    out_models.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data(raw_path)
    df_cleaned, _ = clean_and_deduplicate(df_raw)

    # Save cleaned data
    save_processed_data(df_cleaned, str(PROCESSED_DATA_DIR / "winequality-cleaned.csv"))

    # Train / test split and scaling
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = create_train_test_split(
        df_cleaned, test_size=0.2, random_state=42
    )

    # Save scaled train and test data
    train_export = X_train_scaled.copy()
    train_export["quality_target"] = y_train.values
    save_processed_data(train_export, str(PROCESSED_DATA_DIR / "train_scaled.csv"))

    test_export = X_test_scaled.copy()
    test_export["quality_target"] = y_test.values
    save_processed_data(test_export, str(PROCESSED_DATA_DIR / "test_scaled.csv"))

    results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Select best model by F1 score
    best_name = max(results.keys(), key=lambda k: results[k]["metrics"]["f1"])
    best_model = results[best_name]["model"]

    if save_artifacts:
        joblib.dump(best_model, out_models / "wine_quality_model.joblib")
        joblib.dump(scaler, out_models / "scaler.joblib")

        metadata = {
            "best_model": best_name,
            "feature_names": list(X_train_scaled.columns),
            "metrics": {k: v["metrics"] for k, v in results.items()},
        }
        with open(out_models / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    return {
        "best_model_name": best_name,
        "results": {k: v["metrics"] for k, v in results.items()},
        "feature_names": list(X_train_scaled.columns),
    }


if __name__ == "__main__":
    print("Training models...")
    summary = train_pipeline()
    print(f"Training complete! Best model: {summary['best_model_name']}")
    for model_name, m in summary["results"].items():
        print(f" -> {model_name}: F1={m['f1']:.4f}, ROC-AUC={m['roc_auc']:.4f}, Acc={m['accuracy']:.4f}")
