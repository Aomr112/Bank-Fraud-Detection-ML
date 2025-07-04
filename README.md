# 🏦 Loan Default Prediction using Random Forest

This project predicts whether a customer will repay a loan or default using a supervised machine learning approach based on **Random Forest Classifier**.

---

## 📊 Dataset

- Total Rows: ~246,000
- Selected Features: Top 20 important features (selected using feature importance from Random Forest)
- Target variable: `loan_repaid` (1 = Will repay, 0 = Will default)

---

## 🧠 Problem Type

- Binary classification  
- Imbalanced data, but **no resampling technique** was used (e.g., no SMOTE)

---

## 🔧 Project Pipeline

1. **Preprocessing**
   - `SimpleImputer` for missing values
   - `OrdinalEncoder` for categorical features
   - Used `ColumnTransformer` to apply numeric and categorical pipelines

2. **Feature Selection**
   - Chose top 20 most informative features based on feature importance scores

3. **Model Training**
   - `RandomForestClassifier`
   - Hyperparameter tuning with `RandomizedSearchCV`
   - Key parameters tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`

---

## 🧪 Final Evaluation on Test Set

| Metric     | Score    |
|------------|----------|
| Accuracy   | 0.9909   |
| Precision  | 0.9172   |
| Recall     | 0.9760   |
| F1 Score   | 0.9457   |

> ✅ Model shows **very strong performance** without any oversampling techniques

---

## 📌 Key Features Used

- `AMT_CREDIT`
- `EXT_SOURCE_2`
- `DAYS_EMPLOYED`
- `NAME_CONTRACT_TYPE`
- `AMT_ANNUITY`
- `DAYS_BIRTH`
- ... (list all 20 used if possible)

---

## 🧰 Tools Used

- Python (Jupyter Notebook)
- Scikit-learn
- Pandas, NumPy, Matplotlib

---

## 🚫 Not Used

- No XGBoost (didn’t improve performance)
- No SMOTE or other resampling
- No One-Hot Encoding (used OrdinalEncoder instead)

---

## 💬 Conclusion

Random Forest performed best in this task with excellent generalization. After trying other approaches, Random Forest provided the **best trade-off between performance and simplicity** for this dataset.
