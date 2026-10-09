# 🍷 Red Wine Quality Analytics


An end-to-end machine learning analytics system and interactive web application for analyzing, modeling, and predicting red wine sensory quality from physicochemical tests using the **UCI Machine Learning Repository** Red Wine dataset (Cortez et al., 2009).

---

## 📌 Architecture & Repository Layout

```
red-wine-quality-analytics/
├── .github/
│   └── workflows/
│       └── python-app.yml            # CI: Automated pytest suite across Python 3.10 & 3.11
├── data/
│   ├── raw/
│   │   └── winequality-red.csv       # Original UCI source dataset (read-only, never edited)
│   └── processed/
│       ├── winequality-cleaned.csv   # Deduplicated dataset (1,359 observations)
│       ├── train_scaled.csv          # Stratified training split with standard scaling
│       └── test_scaled.csv           # Stratified test holdout with standard scaling
├── docs/
│   ├── proposal.md                   # Project proposal & milestone specifications
│   ├── final_report.md               # Complete scientific analysis & model report
│   └── figures/                      # Scalable vector graphics and data charts
│       ├── quality_distribution.svg  # Class distribution visualization
│       ├── feature_importance.svg    # Random Forest feature importance chart
│       └── system_architecture.svg   # Dataflow and architecture diagram
├── notebooks/
│   ├── 01_eda_and_visualizations.ipynb      # Exploratory data analysis & statistical distributions
│   ├── 02_preprocessing.ipynb               # Deduplication, feature engineering & scaling
│   └── 03_modeling_and_explainability.ipynb # Benchmarks, ROC curves, confusion matrices
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Dataset ingestion, delimiter parsing & deduplication
│   ├── features.py                   # Domain interactions, target formulation & scalers
│   ├── train.py                      # Multi-model training, evaluation & joblib serialization
│   └── predict.py                    # Inference interface for physicochemical inputs
├── tests/
│   ├── __init__.py
│   ├── test_data.py                  # Unit tests for loaders, transforms & data integrity
│   └── test_model.py                 # Pipeline, evaluation metrics & inference tests
├── app.py                            # Interactive Streamlit dashboard & real-time predictor
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md                         # Project documentation
```

---

## 📊 Dataset Overview

The dataset consists of 1,599 red wine samples evaluated by sensory sommeliers alongside 11 physicochemical measurements:

| Feature | Unit | Description |
| :--- | :--- | :--- |
| `fixed acidity` | $g/dm^3$ | Tartaric acid concentration |
| `volatile acidity` | $g/dm^3$ | Acetic acid concentration (sour/vinegar notes) |
| `citric acid` | $g/dm^3$ | Freshness and fruit flavor enhancer |
| `residual sugar` | $g/dm^3$ | Remaining sugar post-fermentation |
| `chlorides` | $g/dm^3$ | Sodium chloride (salt concentration) |
| `free sulfur dioxide` | $mg/dm^3$ | Free forms of $\text{SO}_2$ preventing oxidation |
| `total sulfur dioxide` | $mg/dm^3$ | Free + bound forms of $\text{SO}_2$ |
| `density` | $g/cm^3$ | Hydrometer density measurement |
| `pH` | scale 0–14 | Acidity scale (wine typical: 3.0–3.8) |
| `sulphates` | $g/dm^3$ | Potassium sulphate contributing to preservation |
| `alcohol` | % vol | Ethanol percentage by volume |
| **`quality`** (target) | score 0–10 | Median expert panel sensory rating (3 to 8) |

### Key Preprocessing Steps
- **Deduplication**: 240 identical records were identified and purged, leaving 1,359 unique rows to prevent cross-validation data leakage.
- **Engineered Domain Features**:
  - `total_acidity`: Combined acidic load ($\text{fixed} + \text{volatile} + \text{citric}$)
  - `bound_sulfur_dioxide`: $\text{total } \text{SO}_2 - \text{free } \text{SO}_2$
  - `free_sulfur_ratio`: $\text{free } \text{SO}_2 / \text{total } \text{SO}_2$ (preservation efficiency)
  - `alcohol_to_sugar_ratio`: Fermentation balance metric

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/red-wine-quality-analytics.git
cd red-wine-quality-analytics
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧪 Pipeline Usage

### Run Data Preprocessing
```bash
python -m src.data_loader
```
Loads raw data from `data/raw/winequality-red.csv`, purges duplicates, and saves `data/processed/winequality-cleaned.csv`.

### Train Models & Save Artifacts
```bash
python -m src.train
```
Trains candidate models (`RandomForest`, `GradientBoosting`, `LogisticRegression`), generates evaluation metrics, selects the best model, and serializes artifacts to `models/`.

### Run Inference
```bash
python -m src.predict
```
Accepts arbitrary wine parameters and returns quality class probabilities and recommendations.

---

## 🖥️ Interactive Web Application (Streamlit)

Launch the interactive wine sensory and quality predictor:

```bash
streamlit run app.py
```

### Features:
- 🎛️ **Physicochemical Sliders**: Real-time evaluation of alcohol, volatile acidity, sulphates, and acidity.
- 🍷 **Preset Profiles**: Instant loading of Premium Reserve, Table Wine, and High-Volatile Acid wine profiles.
- 📊 **Exploratory Visualizations**: Interactive distribution charts and Pearson correlation tables.
- 🔍 **Model Benchmarks**: Accuracy, F1, and feature importance rankings.

---

## 🧪 Running Automated Tests

Run the full unit and integration test suite:

```bash
pytest -v tests/
```

GitHub Actions automatically runs this suite on all pushes and pull requests across **Python 3.10** and **Python 3.11**.

---

## 📈 Modeling Results Summary

| Model | Accuracy | F1-Score ($\ge 7$) | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.886** | **0.649** | **0.892** |
| Gradient Boosting | 0.864 | 0.507 | 0.865 |
| Logistic Regression (Balanced) | 0.743 | 0.443 | 0.804 |

**Dominant Chemical Drivers**:
1. `alcohol` (18.2% relative importance): Premium wines exhibit higher average alcohol concentrations.
2. `sulphates` (12.8%): Essential for oxidative stability.
3. `volatile acidity` (11.4%): Excess acetic acid degrades wine scores significantly.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
