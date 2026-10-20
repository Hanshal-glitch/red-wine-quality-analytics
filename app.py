"""Streamlit Web Application: Red Wine Quality Analytics & Interactive Predictor."""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

from src.data_loader import load_raw_data, clean_and_deduplicate
from src.features import engineer_features
from src.train import train_pipeline
from src.predict import predict_wine_quality, load_artifacts, MODELS_DIR

# Set Page Config
st.set_page_config(
    page_title="Red Wine Quality Analytics",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #722F37;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #fcf8f8;
        border-left: 5px solid #722F37;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #722F37;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_dataset():
    """Load and cache raw and cleaned datasets."""
    raw_df = load_raw_data()
    clean_df, dupes = clean_and_deduplicate(raw_df)
    return raw_df, clean_df, dupes


def ensure_model_artifacts():
    """Ensure trained model artifacts exist, training on the fly if needed."""
    model_path = MODELS_DIR / "wine_quality_model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"
    if not (model_path.exists() and scaler_path.exists() and meta_path.exists()):
        with st.spinner("Training baseline model for initial startup..."):
            train_pipeline()


def main():
    ensure_model_artifacts()
    raw_df, clean_df, dupes_removed = get_dataset()

    # Sidebar Navigation
    st.sidebar.title("🍷 Navigation")
    menu = st.sidebar.radio(
        "Choose Section",
        [
            "Interactive Quality Predictor",
            "Exploratory Data Analysis",
            "Model Performance & Metrics",
            "Project Documentation",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Dataset**: UCI Machine Learning Repository  \n"
        "**Target**: Red Wine Quality (Score 3–8)  \n"
        "**Classification**: High Quality (≥ 7) vs Standard (< 7)"
    )

    # 1. INTERACTIVE PREDICTOR SECTION
    if menu == "Interactive Quality Predictor":
        st.markdown('<div class="main-header">Interactive Wine Quality Predictor</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Adjust physicochemical wine properties to estimate quality score and predict whether a wine achieves premium status (Score ≥ 7).</div>',
            unsafe_allow_html=True,
        )

        # Preset selection
        col_preset, _ = st.columns([2, 3])
        with col_preset:
            preset = st.selectbox(
                "Or load a physicochemical preset profile:",
                ["Custom", "High Quality Reserve (Cabernet/Syrah)", "Typical Table Wine", "Sour/High Volatile Acid Wine"],
            )

        defaults = {
            "fixed acidity": 8.3,
            "volatile acidity": 0.52,
            "citric acid": 0.27,
            "residual sugar": 2.5,
            "chlorides": 0.08,
            "free sulfur dioxide": 14.0,
            "total sulfur dioxide": 46.0,
            "density": 0.9967,
            "pH": 3.31,
            "sulphates": 0.65,
            "alcohol": 10.4,
        }

        if preset == "High Quality Reserve (Cabernet/Syrah)":
            defaults.update({
                "fixed acidity": 10.2,
                "volatile acidity": 0.31,
                "citric acid": 0.49,
                "residual sugar": 2.2,
                "chlorides": 0.075,
                "free sulfur dioxide": 18.0,
                "total sulfur dioxide": 38.0,
                "density": 0.9962,
                "pH": 3.25,
                "sulphates": 0.82,
                "alcohol": 12.5,
            })
        elif preset == "Typical Table Wine":
            defaults.update({
                "fixed acidity": 7.8,
                "volatile acidity": 0.60,
                "citric acid": 0.15,
                "residual sugar": 2.1,
                "chlorides": 0.089,
                "free sulfur dioxide": 12.0,
                "total sulfur dioxide": 50.0,
                "density": 0.9972,
                "pH": 3.36,
                "sulphates": 0.55,
                "alcohol": 9.8,
            })
        elif preset == "Sour/High Volatile Acid Wine":
            defaults.update({
                "fixed acidity": 6.9,
                "volatile acidity": 0.98,
                "citric acid": 0.04,
                "residual sugar": 3.2,
                "chlorides": 0.115,
                "free sulfur dioxide": 8.0,
                "total sulfur dioxide": 65.0,
                "density": 0.9985,
                "pH": 3.52,
                "sulphates": 0.48,
                "alcohol": 9.2,
            })

        st.subheader("Physicochemical Parameters")
        c1, c2, c3 = st.columns(3)

        with c1:
            fixed_acidity = st.slider(
                "Fixed Acidity (g/dm³)", 4.0, 16.0, float(defaults["fixed acidity"]), 0.1,
                help="Tartaric acid concentration responsible for wine freshness.",
                key=f"fixed_acidity_{preset}",
            )
            volatile_acidity = st.slider(
                "Volatile Acidity (g/dm³)", 0.1, 1.6, float(defaults["volatile acidity"]), 0.01,
                help="Acetic acid content; high levels cause an unpleasant vinegar flavor.",
                key=f"volatile_acidity_{preset}",
            )
            citric_acid = st.slider(
                "Citric Acid (g/dm³)", 0.0, 1.0, float(defaults["citric acid"]), 0.01,
                help="Adds crispness and fresh fruit flavors in small quantities.",
                key=f"citric_acid_{preset}",
            )
            residual_sugar = st.slider(
                "Residual Sugar (g/dm³)", 0.5, 16.0, float(defaults["residual sugar"]), 0.1,
                help="Amount of sugar remaining after fermentation ceases.",
                key=f"residual_sugar_{preset}",
            )

        with c2:
            chlorides = st.slider(
                "Chlorides (g/dm³)", 0.01, 0.65, float(defaults["chlorides"]), 0.005,
                help="Amount of mineral salts.",
                key=f"chlorides_{preset}",
            )
            free_so2 = st.slider(
                "Free Sulfur Dioxide (mg/dm³)", 1.0, 75.0, float(defaults["free sulfur dioxide"]), 1.0,
                help="Active SO2 protecting wine against oxidation and microbes.",
                key=f"free_so2_{preset}",
            )
            total_so2 = st.slider(
                "Total Sulfur Dioxide (mg/dm³)", 6.0, 290.0, float(defaults["total sulfur dioxide"]), 1.0,
                help="Free and bound forms of SO2 combined.",
                key=f"total_so2_{preset}",
            )
            density = st.slider(
                "Density (g/cm³)", 0.990, 1.004, float(defaults["density"]), 0.0002,
                help="Correlates with percent alcohol and residual sugar.",
                key=f"density_{preset}",
            )

        with c3:
            ph = st.slider(
                "pH Level", 2.7, 4.1, float(defaults["pH"]), 0.01,
                help="Acidity on a scale of 0 (very acidic) to 14 (basic). Wine usually 3.0-3.8.",
                key=f"ph_{preset}",
            )
            sulphates = st.slider(
                "Sulphates (g/dm³)", 0.3, 2.0, float(defaults["sulphates"]), 0.01,
                help="Potassium sulphate additive contributing to antimicrobial/antioxidant activity.",
                key=f"sulphates_{preset}",
            )
            alcohol = st.slider(
                "Alcohol (% by volume)", 8.0, 15.0, float(defaults["alcohol"]), 0.1,
                help="Percent alcohol content by volume.",
                key=f"alcohol_{preset}",
            )

        # Collect input dictionary
        sample_input = {
            "fixed acidity": fixed_acidity,
            "volatile acidity": volatile_acidity,
            "citric acid": citric_acid,
            "residual sugar": residual_sugar,
            "chlorides": chlorides,
            "free sulfur dioxide": free_so2,
            "total sulfur dioxide": total_so2,
            "density": density,
            "pH": ph,
            "sulphates": sulphates,
            "alcohol": alcohol,
        }

        st.markdown("---")
        st.subheader("Model Prediction Output")

        try:
            res = predict_wine_quality(sample_input)
            prob = res["probability_high_quality"]
            is_premium = res["prediction_binary"] == 1

            p_col1, p_col2 = st.columns([1, 1])

            with p_col1:
                if is_premium:
                    st.success("### 🌟 Premium Quality Wine (Score ≥ 7)")
                    st.write(
                        "The physicochemical profile exhibits favorable balance: elevated alcohol, sufficient sulphates, and controlled volatile acidity."
                    )
                else:
                    st.info("### 🍷 Standard / Table Quality Wine (Score < 7)")
                    st.write(
                        "The physicochemical profile suggests average commercial quality. Consider adjusting volatile acidity or alcohol balance."
                    )

            with p_col2:
                st.metric("High-Quality Probability", f"{res['confidence_percentage']}%")
                st.progress(min(max(float(prob), 0.0), 1.0))

            with st.expander("🔬 Derived Domain Features"):
                sample_df = pd.DataFrame([sample_input])
                eng = engineer_features(sample_df)
                st.dataframe(
                    eng[[
                        "total_acidity",
                        "bound_sulfur_dioxide",
                        "free_sulfur_ratio",
                        "alcohol_to_sugar_ratio",
                    ]].round(3)
                )

        except Exception as e:
            st.error(f"Inference error: {e}")

    # 2. EDA SECTION
    elif menu == "Exploratory Data Analysis":
        st.markdown('<div class="main-header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Inspection of the UCI Red Wine Quality dataset distribution, duplicates, and key correlations.</div>',
            unsafe_allow_html=True,
        )

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Raw Observations", len(raw_df))
        kpi2.metric("Duplicates Removed", dupes_removed)
        kpi3.metric("Clean Observations", len(clean_df))
        kpi4.metric("Features Count", len(clean_df.columns) - 1)

        st.subheader("Quality Distribution")
        dist_df = clean_df["quality"].value_counts().sort_index().reset_index()
        dist_df.columns = ["Quality Score", "Count"]
        st.bar_chart(data=dist_df, x="Quality Score", y="Count")

        st.subheader("Correlation with Quality")
        corr = clean_df.corr(numeric_only=True)["quality"].sort_values(ascending=False).drop("quality")
        corr_df = corr.reset_index()
        corr_df.columns = ["Physicochemical Feature", "Pearson Correlation"]
        st.dataframe(corr_df.style.format({"Pearson Correlation": "{:.3f}"}), use_container_width=True)

        st.subheader("Dataset Preview (Cleaned)")
        st.dataframe(clean_df.head(15), use_container_width=True)

    # 3. MODEL PERFORMANCE SECTION
    elif menu == "Model Performance & Metrics":
        st.markdown('<div class="main-header">Model Performance & Benchmarks</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Comparison across trained classification pipelines on the stratified test set.</div>',
            unsafe_allow_html=True,
        )

        meta_path = MODELS_DIR / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)

            st.write(f"**Best Performing Model**: `{metadata.get('best_model')}`")

            metrics_data = []
            for name, m in metadata.get("metrics", {}).items():
                metrics_data.append({
                    "Model": name,
                    "Accuracy": round(m["accuracy"], 3),
                    "Precision": round(m["precision"], 3),
                    "Recall": round(m["recall"], 3),
                    "F1 Score": round(m["f1"], 3),
                    "ROC-AUC": round(m["roc_auc"], 3),
                })
            st.table(pd.DataFrame(metrics_data))

            # Feature Importance
            model, scaler, _ = load_artifacts()
            feat_names = metadata.get("feature_names", [])
            if hasattr(model, "feature_importances_") and len(feat_names) == len(model.feature_importances_):
                st.subheader("Feature Importances")
                fi_df = pd.DataFrame({
                    "Feature": feat_names,
                    "Importance": model.feature_importances_,
                }).sort_values(by="Importance", ascending=False)
                st.bar_chart(fi_df.set_index("Feature"))
        else:
            st.warning("Model metadata not found. Run `src/train.py` to generate metrics.")

    # 4. DOCUMENTATION SECTION
    elif menu == "Project Documentation":
        st.markdown('<div class="main-header">Project Architecture & Methodology</div>', unsafe_allow_html=True)
        st.markdown(
            """
            ### System Overview
            This repository provides an end-to-end machine learning and analytics workflow for the **UCI Red Wine Quality** dataset.
            
            #### Key Modules:
            - **`data/raw/`**: Holds untouched original UCI CSV file (`winequality-red.csv`).
            - **`data/processed/`**: Houses deduplicated (`1,359` unique rows) and scaled datasets (`train_scaled.csv`, `test_scaled.csv`).
            - **`src/data_loader.py`**: Clean ingestion, delimiter handling, deduplication, and persistence.
            - **`src/features.py`**: Domain-specific feature engineering (acidity sums, SO2 ratios, alcohol-sugar ratio) & scaling.
            - **`src/train.py`**: Trains RandomForest, GradientBoosting, and LogisticRegression with stratified evaluation.
            - **`src/predict.py`**: High-performance inference module with serialized model artifacts.
            - **`tests/`**: Unit test suite (`test_data.py`, `test_model.py`) integrated with GitHub Actions CI.
            - **`notebooks/`**: Guided Jupyter notebooks covering EDA, Preprocessing, and Explainability.
            """
        )


if __name__ == "__main__":
    main()
