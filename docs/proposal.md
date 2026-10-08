# Project Proposal: Red Wine Quality Analytics & Machine Learning Modeling

## 1. Executive Summary
Wine quality assessment has traditionally depended heavily on human sensory evaluation (organoleptic testing) conducted by certified sommeliers and enologists. While sensory testing is valuable, it is inherently subjective, prone to taster fatigue, and difficult to scale across industrial winemaking processes. 

This project establishes a reproducible, automated machine learning pipeline to model, predict, and analyze red wine quality based purely on objective physicochemical attributes. Using the benchmark **UCI Red Wine Quality** dataset, we develop analytical reports, clean data pipelines, interpretable machine learning models, and an interactive prediction application.

---

## 2. Problem Statement & Motivation
Winemakers require rapid, cost-effective methods to gauge product quality and spot chemical imbalances during fermentation, barrel aging, and bottling. Identifying high-leverage chemical markers (such as optimal volatile acidity thresholds, free/total sulfur dioxide balance, and alcohol ratios) allows producers to:
- Optimize fermentation recipes and monitoring protocols.
- Intervene early if volatile acidity spikes (preventing spoilage).
- Standardize quality assurance classifications prior to sensory panel reviews.

---

## 3. Dataset Characteristics
- **Source**: UCI Machine Learning Repository / Cortez et al. (2009).
- **Instances**: 1,599 red wine samples (Portuguese "Vinho Verde" variants).
- **Input Features (Physicochemical Tests)**:
  1. `fixed acidity`: Tartaric acid concentration ($g/dm^3$)
  2. `volatile acidity`: Acetic acid concentration ($g/dm^3$)
  3. `citric acid`: Freshness/crispness acid ($g/dm^3$)
  4. `residual sugar`: Remaining sugar post-fermentation ($g/dm^3$)
  5. `chlorides`: Mineral salt content ($g/dm^3$)
  6. `free sulfur dioxide`: Active antioxidant/antimicrobial agent ($mg/dm^3$)
  7. `total sulfur dioxide`: Bound and free sulfur dioxide ($mg/dm^3$)
  8. `density`: Hydrometer density measurement ($g/cm^3$)
  9. `pH`: Acid-base equilibrium (log scale)
  10. `sulphates`: Potassium sulphate additive ($g/dm^3$)
  11. `alcohol`: Ethanol percentage by volume (% vol)
- **Target Variable**:
  - `quality`: Median score from at least three sensory evaluators, scaled between 0 (very bad) and 10 (excellent).

---

## 4. Methodology & Technical Approach

```
+-------------------------------------------------------+
|                    Raw Data Ingestion                 |
|               (data/raw/winequality-red.csv)          |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|             Preprocessing & Deduplication             |
|   - Deduplicate rows (1,599 -> 1,359 unique rows)     |
|   - Check for missing values / data type integrity   |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|                  Feature Engineering                  |
|   - Total Acidity = Fixed + Volatile + Citric         |
|   - Bound SO2 = Total SO2 - Free SO2                  |
|   - Free SO2 Ratio = Free SO2 / Total SO2             |
|   - Alcohol-to-Sugar Ratio                            |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|           Train / Test Stratified Split & Scaling     |
|   - 80/20 Stratified Split                            |
|   - Standard Scaling fit strictly on training set     |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|                  Model Benchmarking                   |
|   - Random Forest Classifier (Champion)               |
|   - Gradient Boosting Classifier                      |
|   - Regularized Logistic Regression                   |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|            Evaluation & Explainability                |
|   - Accuracy, Precision, Recall, F1, ROC-AUC          |
|   - Confusion Matrix & Gini Feature Importance        |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|              Production & Delivery                    |
|   - Serialized Artifacts (.joblib)                    |
|   - Automated CI (GitHub Actions)                     |
|   - Interactive Streamlit Dashboard (app.py)          |
+-------------------------------------------------------+
```

---

## 5. Timeline & Milestones
| Phase | Milestone | Deliverable |
| :--- | :--- | :--- |
| **Phase 1** | Data Acquisition & Exploratory Analysis | `01_eda_and_visualizations.ipynb`, `data/raw/` |
| **Phase 2** | Data Cleaning & Feature Engineering | `src/data_loader.py`, `src/features.py`, `02_preprocessing.ipynb` |
| **Phase 3** | Model Benchmarking & Tuning | `src/train.py`, `03_modeling_and_explainability.ipynb` |
| **Phase 4** | Verification, CI & Inference Service | `tests/`, `.github/workflows/python-app.yml`, `src/predict.py` |
| **Phase 5** | Interactive UI & Final Documentation | `app.py`, `docs/final_report.md`, `README.md` |

---

## 6. Success Criteria
1. **Reproducibility**: Clear modular structure (`src/`, `tests/`, `notebooks/`) with zero data leakage.
2. **Quality Metric**: Champion model achieving ROC-AUC $\ge 0.85$ and F1-Score $\ge 0.60$ for high-quality classification on the deduplicated test holdout.
3. **CI Automation**: 100% test pass rate in GitHub Actions on Python 3.10 and 3.11.
