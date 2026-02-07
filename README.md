# Credit Card Fraud Detection
An end-to-end machine learning project to detect fraudulent credit card transactions using imbalanced classification techniques.

## Dataset
This project uses the **Credit Card Fraud Detection** dataset available on Kaggle.

🔗 Dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains anonymized credit card transactions made by European cardholders in September 2013.
Due to privacy concerns, the features are PCA-transformed (V1–V28).

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM
- SHAP

### Key Insights from EDA
- Dataset is highly imbalanced (0.17% fraud)
- Accuracy is misleading as a metric
- Fraud transactions tend to have smaller amounts
- Temporal patterns exist in fraudulent activity
- Feature space is PCA-transformed for privacy

### Baseline Model Observations
- Logistic Regression achieves high accuracy due to class imbalance
- Fraud recall is extremely low, making the model unusable in practice
- ROC-AUC appears acceptable but hides poor minority class performance
- Precision-Recall curve reveals the true weakness of the model

### Imbalance Handling Observations
- Class weighting improves recall without altering data distribution
- SMOTE significantly increases fraud recall
- Precision decreases as recall increases, requiring business-driven threshold selection
- Threshold tuning is more impactful than model choice

### Advanced Model Selection
Tree-based models significantly outperform linear baselines on imbalanced fraud data.
XGBoost achieved higher precision at the same recall compared to Logistic Regression and Random Forest.

### Model Explainability using SHAP
SHAP values were used to interpret both global feature importance and individual predictions.
The model identifies fraud based on complex interactions among PCA-transformed features.

## Demo Application
A Streamlit-based web application was built to simulate real-time fraud detection.
The system outputs fraud probability and applies business-driven thresholds to decide whether a transaction should be allowed, reviewed, or blocked.

