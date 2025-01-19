import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

# Handle missing values 
train = train.dropna()

# Transform date column
def transform_date(df, col):
    df[col] = pd.to_datetime(df[col])
    df['year'] = df[col].dt.year.astype('float64')
    df['quarter'] = df[col].dt.quarter.astype('float64')
    df['month'] = df[col].dt.month.astype('float64')
    df['day'] = df[col].dt.day.astype('float64')
    df['day_of_week'] = df[col].dt.dayofweek.astype('float64')
    return df

train = transform_date(train, 'date')
test = transform_date(test, 'date')

# Drop the original 'date' column
train = train.drop(columns=['date'], axis=1)
test = test.drop(columns=['date'], axis=1)

# One-hot encode categorical features
cat_cols = ['country', 'store', 'product']
train = pd.get_dummies(train, columns=cat_cols)
test = pd.get_dummies(test, columns=cat_cols)

# Apply log transformation to target variable
train['num_sold'] = np.log1p(train['num_sold'])

# Split train data into features and target
X = train.drop(columns=['num_sold'])
y = train['num_sold']

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize global best MAPEs
best_mape_lgb = float('inf')
best_mape_xgb = float('inf')
best_mape_rf = float('inf')

# Define objective functions for each model
def objective_lgb(trial):
    global best_mape_lgb
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 2000, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 10, 16),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 0.1),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'objective': 'regression',
        'metric': 'mape',
        'device': 'gpu',
        'n_jobs': -1
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    mape = mean_absolute_percentage_error(y_valid, y_pred)
    if mape < best_mape_lgb:
        best_mape_lgb = mape
        joblib.dump(model, 'best_model_lgb.pkl')
    return mape

def objective_xgb(trial):
    global best_mape_xgb
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 2000, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 16),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    mape = mean_absolute_percentage_error(y_valid, y_pred)
    if mape < best_mape_xgb:
        best_mape_xgb = mape
        joblib.dump(model, 'best_model_xgb.pkl')
    return mape

def objective_rf(trial):
    global best_mape_rf
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 16),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'n_jobs': -1,
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    mape = mean_absolute_percentage_error(y_valid, y_pred)
    if mape < best_mape_rf:
        best_mape_rf = mape
        joblib.dump(model, 'best_model_rf.pkl')
    return mape

# Optimize each model
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=10)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=10)

study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=10)

# Load the best models
lgb_model = joblib.load('best_model_lgb.pkl')
xgb_model = joblib.load('best_model_xgb.pkl')
rf_model = joblib.load('best_model_rf.pkl')

# Calculate weights based on MAPE
inverse_mape_sum = (1/best_mape_lgb) + (1/best_mape_xgb) + (1/best_mape_rf)
weight_lgb = (1/best_mape_lgb) / inverse_mape_sum
weight_xgb = (1/best_mape_xgb) / inverse_mape_sum
weight_rf = (1/best_mape_rf) / inverse_mape_sum

print(f"Ensemble Weights: LightGBM={weight_lgb}, XGBoost={weight_xgb}, RandomForest={weight_rf}")

# Generate predictions for the test set
pred_lgb = np.expm1(lgb_model.predict(test))
pred_xgb = np.expm1(xgb_model.predict(test))
pred_rf = np.expm1(rf_model.predict(test))

# Weighted ensemble
final_predictions = (weight_lgb * pred_lgb) + (weight_xgb * pred_xgb) + (weight_rf * pred_rf)

# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'num_sold': final_predictions
})

submission.to_csv('submission.csv', index=False)
print("submission.csv saved.")