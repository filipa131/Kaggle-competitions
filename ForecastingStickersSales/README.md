# Kaggle Competition: Playground Series - Season 5, Episode 1 (S5E1)

## Overview
This repository contains the solution for the [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s5e1), where the goal is to predict the number of products sold (`num_sold`). The final model achieved a **MAPE of 0.09729** using a weighted ensemble of LightGBM, XGBoost, and Random Forest.

## Approach
1. **Data Preprocessing**:
   - Extracted date features (year, month, day, etc.).
   - One-hot encoded categorical columns (`country`, `store`, `product`).
   - Log-transformed the target variable for normalization.

2. **Model Training**:
   - Hyperparameter tuning using **Optuna** for:
     - LightGBM
     - XGBoost
     - Random Forest
   - Saved the best models using `joblib`.

3. **Ensemble**:
   - Combined predictions from the three models using weights inversely proportional to their validation MAPE.

4. **Submission**:
   - Generated predictions for the test set and created `submission.csv`.

## Results
- **Final Validation MAPE**: 0.09729