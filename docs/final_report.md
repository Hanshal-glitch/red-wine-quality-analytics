# Final Report: Red Wine Quality Analytics & Predictive Modeling

**Authors:** Red Wine Quality Analytics Team  
**Dataset:** Cortez et al., 2009 (UCI Machine Learning Repository)  
**Status:** Complete  

---

## 1. Executive Summary
This report summarizes the findings, modeling methodology, and production deployment of the **Red Wine Quality Analytics** project. The primary goal was to construct an automated analytical system capable of predicting wine quality and uncovering the primary chemical drivers distinguishing premium wines from standard commercial batches.

Key outcomes:
- **Deduplicated Dataset**: Identified and removed 240 duplicate records, training models on 1,359 unique physicochemical profiles.
- **Top Predictive Features**: **Alcohol content** (+ correlation 0.48), **Sulphates** (+ correlation 0.25), and **Volatile Acidity** (- correlation -0.39) are the dominant determinants of wine quality.
- **Champion Model**: A balanced **Random Forest Classifier** achieved an **ROC-AUC of ~0.89** and an **F1-score of ~0.64** on the held-out test set for identifying high-quality wines ($\text{score} \ge 7$).
- **End-to-End Pipeline**: Includes automated ingestion, scikit-learn feature transformers, serialization, continuous integration via GitHub Actions, and an interactive **Streamlit** dashboard.

---

## 2. Exploratory Data Analysis (EDA) Highlights

### 2.1 Distribution of Quality Ratings
The sensory quality ratings in the dataset range from 3 to 8 (mean: 5.64, median: 6). Extreme scores (3, 4, 8) are relatively rare, while intermediate ratings (5 and 6) represent approximately 82% of all samples.

| Quality Score | Raw Count | Deduplicated Count | Category |
| :---: | :---: | :---: | :---: |
| 3 | 10 | 10 | Poor |
| 4 | 53 | 53 | Below Average |
| 5 | 681 | 577 | Average |
| 6 | 638 | 535 | Above Average |
| 7 | 199 | 167 | High Quality / Premium |
| 8 | 18 | 17 | Exceptional |

### 2.2 Key Physicochemical Correlations
Pearson correlation analysis reveals strong relationships with the sensory quality metric:
1. **Alcohol ($\rho = +0.476$)**: Positively associated with perceived richness, body, and flavor complexity.
2. **Volatile Acidity ($\rho = -0.391$)**: Acetic acid concentration. Elevated levels create an undesirable vinegar taste and harsh aroma, strongly depressing quality scores.
3. **Sulphates ($\rho = +0.251$)**: Antimicrobial and antioxidant properties preserve fruit character and prevent oxidation.
4. **Citric Acid ($\rho = +0.226$)**: Contributes to palate freshness and crisp acidity.
5. **Density ($\rho = -0.175$)**: Higher density often correlates with lower alcohol content and unfermented sugar.

---

## 3. Preprocessing & Feature Engineering

### 3.1 Data Deduplication
The original raw dataset contained 240 identical records (duplicate sampling or repeated testing). Retaining duplicates artificially inflates cross-validation performance through data leakage between folds. All duplicates were filtered out, leaving 1,359 unique rows.

### 3.2 Domain-Derived Feature Engineering
In addition to the 11 baseline attributes, four domain-specific interaction features were engineered:
- **Total Acidity**: $\text{Fixed Acidity} + \text{Volatile Acidity} + \text{Citric Acid}$.
- **Bound Sulfur Dioxide**: $\max(0, \text{Total SO}_2 - \text{Free SO}_2)$, measuring the portion of preservative bound to carbonyl compounds.
- **Free Sulfur Ratio**: $\frac{\text{Free SO}_2}{\text{Total SO}_2 + 10^{-5}}$, quantifying the percentage of active sulfur dioxide available for oxidation protection.
- **Alcohol-to-Sugar Ratio**: $\frac{\text{Alcohol}}{\text{Residual Sugar} + 10^{-5}}$, quantifying fermentation completeness and perceived sweetness dryness.

### 3.3 Target Formulation & Scaling
To solve the commercial problem of identifying premium wines suitable for premium labeling and pricing, the target was modeled as a binary classification:
$$\text{Class} = \begin{cases} 1 \ (\text{High Quality / Premium}) & \text{if } \text{quality} \ge 7 \\ 0 \ (\text{Standard / Normal}) & \text{if } \text{quality} < 7 \end{cases}$$

Features were standardized using `StandardScaler` fitted strictly on the 80% training split to eliminate data leakage.

---

## 4. Model Benchmarking & Performance

Candidate models were trained on the stratified training set ($N=1,087$) and evaluated on the independent test set ($N=272$):

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **0.886** | **0.625** | **0.676** | **0.649** | **0.892** |
| Gradient Boosting Classifier | 0.864 | 0.529 | 0.486 | 0.507 | 0.865 |
| Logistic Regression (Balanced) | 0.743 | 0.317 | 0.730 | 0.443 | 0.804 |

### Observations:
- **Random Forest** achieved the strongest balance of precision and recall on the positive minority class (F1 = 0.649) while delivering high discriminative power (ROC-AUC = 0.892).
- **Logistic Regression** achieved high recall (0.730) due to class weighting, but suffered from excessive false positives.

---

## 5. Model Explainability & Feature Importance

Feature importance extraction from the champion Random Forest revealed:
1. **Alcohol** (~18.2% relative importance): Strongest single separator between table wines and premium vintages.
2. **Sulphates** (~12.8%): Critical in preserving sensory stability.
3. **Volatile Acidity** (~11.4%): Immediate indicator of spoilage if exceeding $0.65\ g/dm^3$.
4. **Total Acidity & Citric Acid** (~10.1%): Balance pH and overall freshness.
5. **Free Sulfur Ratio** (~8.7%): Confirms that antioxidant protection efficiency improves shelf-stability and tasting notes.

---

## 6. Software Architecture & CI Pipeline

```
red-wine-quality-analytics/
|-- .github/workflows/python-app.yml   # Multi-version CI testing (Python 3.10, 3.11)
|-- data/
|   |-- raw/winequality-red.csv        # Read-only UCI source data
|   |-- processed/                     # Cleaned and scaled train/test splits
|-- docs/
|   |-- proposal.md                    # Project proposal & timeline
|   |-- final_report.md                # Comprehensive final analysis report
|   |-- figures/                       # Generated visual assets
|-- models/
|   |-- wine_quality_model.joblib      # Serialized champion model artifact
|   |-- scaler.joblib                  # Serialized fitted StandardScaler
|   |-- model_metadata.json            # Model benchmarks & feature metadata
|-- notebooks/
|   |-- 01_eda_and_visualizations.ipynb
|   |-- 02_preprocessing.ipynb
|   |-- 03_modeling_and_explainability.ipynb
|-- src/
|   |-- data_loader.py                 # Ingestion & deduplication
|   |-- features.py                    # Domain transformations & scaling
|   |-- train.py                       # Training pipeline & evaluation
|   |-- predict.py                     # Inference interface
|-- tests/
|   |-- test_data.py                   # Data integrity and loader tests
|   |-- test_model.py                  # Pipeline and prediction tests
|-- app.py                             # Interactive Streamlit application
|-- pytest.ini                         # Pytest configuration & python path root
|-- requirements.txt                   # Dependency specifications
|-- LICENSE                            # MIT License
|-- README.md                          # Repository guide
```

---

## 7. Conclusions & Winemaking Recommendations
1. **Maintain Controlled Volatile Acidity**: Keep volatile acidity strictly below $0.50\ g/dm^3$ to avoid negative sensory penalties.
2. **Optimize Alcohol Fermentation**: High-quality classifications strongly favor wines with alcohol content in the 11.5%–13.0% range.
3. **Calibrate Sulphate Additions**: Sufficient sulphates ($0.70 - 0.85\ g/dm^3$) provide oxidative buffering essential for premium perception.
4. **Interactive Dashboard**: The deployed Streamlit interface allows cellar masters and quality managers to test chemical formulations before bottling.
