# 🏠 House Price Prediction — End-to-End Machine Learning Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/allulu989/house-price-prediction-ml/blob/main/house_price_prediction.ipynb)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-House%20Prices-20BEFF)

Predicting residential house prices from 79 property features using a full machine learning workflow: exploratory analysis, feature engineering, model comparison, hyperparameter tuning, error analysis, and a Kaggle-ready submission.

---

## 📌 Problem

Given details about a house (size, quality, age, location, condition), **predict its final sale price**.

- **Dataset:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (1,460 training houses, 1,459 test houses, Ames, Iowa)
- **Metric:** RMSE on `log(SalePrice)`, the official Kaggle metric, so errors on cheap and expensive homes are weighted equally

---

## 📊 Results

| Model | 5-Fold CV RMSE (log) |
|---|---|
| Random Forest | 0.1359 |
| Linear Regression | 0.1346 |
| Gradient Boosting (default) | 0.1251 |
| Tuned Gradient Boosting | 0.1132 |
| Tuned Lasso | 0.1116 |
| **Blend: Gradient Boosting + Lasso** | **0.1083** ✅ |

**Final model on an unseen hold-out set (20% of the data):**

| Metric | Value |
|---|---|
| R² | **0.939** |
| Mean Absolute Error | **~$13,300** |
| Mean Absolute % Error | **~8.1%** |

<p align="center">
  <img src="images/model_comparison.png" width="48%">
  <img src="images/predicted_vs_actual.png" width="48%">
</p>

---

## 💡 Key Insights

- **Size and quality drive price.** The engineered `TotalSF` feature and `OverallQual` are by far the strongest predictors.
- **Feature engineering paid off.** Three of the top five features (`TotalSF`, `TotalBath`, `HouseAge`) were created in this project.
- **Log transform + scaling matter.** With them, regularized linear models (Lasso, Ridge) beat tree-based models out of the box.
- **Blending helps.** Lasso captures linear trends and Gradient Boosting captures non-linear interactions, so averaging them reduced the error further.

<p align="center">
  <img src="images/feature_importance.png" width="60%">
</p>

---

## 🔄 Workflow

```
Load data → EDA → Cleaning & Feature Engineering → Preprocessing Pipeline
→ Model Comparison (5-Fold CV) → Hyperparameter Tuning → Blending
→ Hold-out Evaluation & Error Analysis → Feature Importance → Kaggle Submission
```

**1. Exploratory Data Analysis**
- Target distribution: prices are right-skewed (skew ≈ 1.88), so the model is trained on `log(SalePrice)`
- Missing values: most mean the feature does not exist (no pool, no garage), not missing data
- Correlation analysis and outlier detection

<p align="center">
  <img src="images/target_distribution.png" width="70%">
</p>

**2. Cleaning & Feature Engineering**
- Removed 2 extreme outliers (very large houses sold at unusually low prices)
- Converted `MSSubClass` from number to category
- Created new features: `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, `TotalPorch`

**3. Preprocessing (scikit-learn `Pipeline` + `ColumnTransformer`)**
- Numeric: median imputation → standard scaling
- Categorical: `"None"` imputation → one-hot encoding
- All steps are fitted inside cross-validation folds to **prevent data leakage**

**4. Modeling & Tuning**
- Compared 6 models: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting
- `RandomizedSearchCV` for Gradient Boosting, alpha search for Lasso
- Final model: average of tuned Gradient Boosting and tuned Lasso

**5. Error Analysis**
- Largest errors come from **low-priced homes in older neighborhoods** and **abnormal sales** (e.g. foreclosures), where price depends on circumstances not fully captured in the data

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Google Colab`

---

## ▶️ How to Run

1. Click the **Open in Colab** badge at the top
2. Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
3. In Colab: `Runtime` → `Run all`, then upload the two CSV files when prompted
4. The notebook produces `submission.csv`, ready to submit on Kaggle

> The dataset is not included in this repository. Please download it from Kaggle.

---

## 📁 Project Structure

```
house-price-prediction-ml/
├── house_price_prediction.ipynb   # Full analysis and modeling notebook
├── images/                        # Plots used in this README
├── requirements.txt
└── README.md
```

---

## 🚀 Next Steps

- Try XGBoost, LightGBM, and CatBoost with a stacking ensemble
- Apply log / Box-Cox transforms to other skewed numeric features
- Add neighborhood-level price features (computed inside CV to avoid leakage)

---

## 👩‍💻 Author

**Allulu Alkheraiji** — Information Technology graduate

[LinkedIn](https://www.linkedin.com/in/allulu-alkheraiji-9a6834273) · [Portfolio](https://allulu989.github.io/portflio-2026/)
