"""
Model Training for House Price Prediction

This script trains and evaluates multiple regression models on the preprocessed house price data.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# File paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'preprocessing')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Load preprocessed data
def load_preprocessed_data():
    """Load the preprocessed data."""
    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val

# Evaluation metrics
def evaluate_model(y_true, y_pred, model_name):
    """Calculate and print evaluation metrics."""
    # Convert back from log scale
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Print metrics
    print(f"\n{model_name} - Evaluation Metrics:")
    print(f"  RMSE (log scale): {rmse:.4f}")
    print(f"  MAE (log scale): {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"\n  RMSE (original scale): ${np.expm1(rmse):,.2f}")
    
    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Plot feature importance
def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[-top_n:]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {model_name}')
        plt.barh(range(len(indices)), importance[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(MODEL_DIR, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Feature importance plot saved to {plot_path}")
    else:
        print(f"Model {model_name} does not support feature importance visualization.")

# Train and evaluate models
def train_and_evaluate_models(X_train, X_val, y_train, y_val, feature_names):
    """Train and evaluate multiple regression models."""
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='rmse'),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Evaluate on training set
        print(f"\n{name} - Training Set:")
        train_metrics = evaluate_model(y_train, y_pred_train, "Training Set")
        
        # Evaluate on validation set
        print(f"\n{name} - Validation Set:")
        val_metrics = evaluate_model(y_val, y_pred_val, "Validation Set")
        
        # Save the model
        model_path = os.path.join(MODEL_DIR, f'{name.lower().replace(" ", "_")}.joblib')
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Plot feature importance for tree-based models
        if name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']:
            plot_feature_importance(model, feature_names, name)
        
        # Store results
        results.append({
            'model': name,
            'train_rmse': train_metrics['rmse'],
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae'],
            'val_r2': val_metrics['r2']
        })
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_rmse')
    results_file = os.path.join(MODEL_DIR, 'model_comparison.csv')
    results_df.to_csv(results_file, index=False)
    
    print("\nModel comparison saved to:", results_file)
    return results_df

def main():
    """Main function to run model training and evaluation."""
    # Load preprocessed data
    X_train, X_val, y_train, y_val = load_preprocessed_data()
    
    # Load feature names
    feature_names = []
    feature_names_file = os.path.join(DATA_DIR, 'feature_names.txt')
    if os.path.exists(feature_names_file):
        with open(feature_names_file, 'r') as f:
            feature_names = [line.strip() for line in f if line.strip()]
    else:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_val, y_train, y_val, feature_names)
    
    # Print results
    print("\n" + "="*50)
    print("Model Comparison (sorted by Validation RMSE):")
    print("-"*50)
    print(results[['model', 'val_rmse', 'val_mae', 'val_r2']].to_string(index=False))
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
