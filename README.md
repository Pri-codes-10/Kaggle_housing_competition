# Kaggle_housing_competition
This notebook has provided a rank of 217 out of 4000 teams
<img width="1545" height="812" alt="image" src="https://github.com/user-attachments/assets/e1085b28-1a94-4d7d-b6ac-2001bc3f198b" />
# House Prices: Advanced Regression Techniques (Kaggle Solution)

A robust, multi-model ensemble pipeline built for the Kaggle **"House Prices: Advanced Regression Techniques"** competition. This pipeline achieves a solid leaderboard score of **0.12007** using out-of-fold cross-validation and a weighted blend of advanced gradient boosters and regularized linear models.

---

## 🚀 Pipeline Architecture

The solution uses a 5-fold cross-validation strategy combined with a weighted blend of five distinct models to minimize variance and generalization error:

1. **XGBoost Regressor**: Optimized gradient boosting with hyperparameter fine-tuning (`max_depth=4`, `learning_rate=0.012`).
2. **LightGBM Regressor**: Fast, leaf-wise tree growth focusing on robust split patterns.
3. **LassoCV**: Linear model with L1 regularization via cross-validation and robust scaling.
4. **RidgeCV**: Linear model with L2 regularization to stabilize coefficient weights.
5. **ElasticNetCV**: Combined L1 and L2 regularization for handling multicollinearity among features.

---

## 🛠️ Key Technical Steps

* **Outlier Handling**: Strict removal of influential high-square-footage, low-price outliers (`GrLivArea > 4000` / `SalePrice < 300000`).
* **Missing Value Imputation**: Handled categorical missingness (e.g., assigning `"None"` to structural features where absence means none) and numeric zero-fill or median imputation.
* **Feature Engineering**: Generated high-value interaction and composite features:
  * `TotalSF` (Combined basement and first/second floor areas)
  * `TotalBath` (Weighted sum of full and half bathrooms)
  * `HouseAge` & `RemodAge` (Property age metrics based on sale year)
  * Quality interaction terms (e.g., `OverallQual * TotalSF`)
* **Skew Correction**: Automatically detected numeric features with high positive skew (`> 0.75`) and applied `log1p` transformation.
* **Encoding**: Applied one-hot encoding (`pd.get_dummies`) aligned safely across train and test sets.

---

## 📦 Prerequisites & Dependencies

To run this pipeline, ensure you have the following Python libraries installed:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm scipy
