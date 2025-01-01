# Santander Customer Transaction Prediction Analysis and Modeling

Competition link: https://www.kaggle.com/competitions/santander-customer-transaction-prediction

This project explores and models the Santander Customer Transaction Prediction dataset with the following steps:

1. **Data Loading**: Load train and test datasets using pandas.

2. **Exploratory Data Analysis (EDA)**:
   - Dataset structure, summary statistics, and missing values.
   - Target variable distribution and feature density plots.

3. **Feature Analysis**:
   - Correlation heatmap.
   - Advanced statistical measures like skewness, kurtosis, and outlier visualization.

4. **Machine Learning Model**:
   - **Preprocessing**: Standardize features, apply PCA, and handle class imbalance with SMOTE.
   - **Modeling**: Use StratifiedKFold and RandomizedSearchCV for hyperparameter tuning of XGBoost, LightGBM, and CatBoost.
   - **Ensemble**: Combine models via weighted soft voting.

**Outcome**: Optimized ensemble model trained and tested with predictions saved for further use.

Achieved public score for this competition: 0.85378