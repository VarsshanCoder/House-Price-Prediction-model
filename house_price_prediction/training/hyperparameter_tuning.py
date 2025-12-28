"""
Hyperparameter Tuning for House Price Prediction Models

This script performs hyperparameter tuning for the top-performing models
from the initial model training phase.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

# File paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'preprocessing')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Custom scorer for RMSE (lower is better)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Load preprocessed data
def load_preprocessed_data():
    """Load the preprocessed data."""
    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    
    print(f"Training data shape: {X_train.shape}")
    
    return X_train, y_train

# Hyperparameter tuning for Ridge
def tune_ridge(X, y):
    """Tune hyperparameters for Ridge regression."""
    print("\nTuning Ridge regression...")
    
    # Define parameter grid
    param_grid = {
        'alpha': np.logspace(-3, 3, 20),  # Regularization strength
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
    
    # Initialize model
    model = Ridge(random_state=42)
    
    # Randomized search
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings that are sampled
        scoring={'rmse': make_scorer(rmse, greater_is_better=False), 'r2': 'r2'},
        refit='rmse',
        cv=kfold,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    # Fit the model
    random_search.fit(X, y)
    
    # Print results
    print("Best parameters:", random_search.best_params_)
    print("Best RMSE:", -random_search.best_score_)  # Negative because of greater_is_better=False
    
    # Save the best model
    best_model = random_search.best_estimator_
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'ridge_tuned.joblib'))
    
    return best_model

# Hyperparameter tuning for Gradient Boosting
def tune_gradient_boosting(X, y):
    """Tune hyperparameters for Gradient Boosting."""
    print("\nTuning Gradient Boosting...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None],
        'random_state': [42]
    }
    
    # Initialize model
    model = GradientBoostingRegressor()
    
    # Randomized search
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings that are sampled
        scoring={'rmse': make_scorer(rmse, greater_is_better=False), 'r2': 'r2'},
        refit='rmse',
        cv=kfold,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    # Fit the model
    random_search.fit(X, y)
    
    # Print results
    print("Best parameters:", random_search.best_params_)
    print("Best RMSE:", -random_search.best_score_)  # Negative because of greater_is_better=False
    
    # Save the best model
    best_model = random_search.best_estimator_
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'gradient_boosting_tuned.joblib'))
    
    return best_model

# Hyperparameter tuning for LightGBM
def tune_lightgbm(X, y):
    """Tune hyperparameters for LightGBM."""
    print("\nTuning LightGBM...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6, -1],  # -1 means no limit
        'num_leaves': [31, 50, 100],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1],
        'random_state': [42]
    }
    
    # Initialize model
    model = LGBMRegressor()
    
    # Randomized search
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings that are sampled
        scoring={'rmse': make_scorer(rmse, greater_is_better=False), 'r2': 'r2'},
        refit='rmse',
        cv=kfold,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    # Fit the model
    random_search.fit(X, y)
    
    # Print results
    print("Best parameters:", random_search.best_params_)
    print("Best RMSE:", -random_search.best_score_)  # Negative because of greater_is_better=False
    
    # Save the best model
    best_model = random_search.best_estimator_
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'lightgbm_tuned.joblib'))
    
    return best_model

def main():
    """Main function to run hyperparameter tuning."""
    # Load preprocessed data
    X, y = load_preprocessed_data()
    
    # Tune models
    print("\n" + "="*50)
    print("Starting Hyperparameter Tuning")
    print("="*50)
    
    # Tune Ridge
    print("\n" + "="*50)
    print("Tuning Ridge Regression")
    print("="*50)
    ridge_best = tune_ridge(X, y)
    
    # Tune Gradient Boosting
    print("\n" + "="*50)
    print("Tuning Gradient Boosting")
    print("="*50)
    gb_best = tune_gradient_boosting(X, y)
    
    # Tune LightGBM
    print("\n" + "="*50)
    print("Tuning LightGBM")
    print("="*50)
    lgbm_best = tune_lightgbm(X, y)
    
    print("\nHyperparameter tuning complete!")
    print(f"Best models saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
