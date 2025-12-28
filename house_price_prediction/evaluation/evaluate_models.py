"""
Model Evaluation Script for House Price Prediction

This script evaluates the performance of the trained models on the test set.
It also generates comparison metrics and visualizations.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# File paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'preprocessing')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'evaluation')
os.makedirs(EVAL_DIR, exist_ok=True)

# Custom evaluation metrics
def evaluate_model(y_true, y_pred, model_name, set_name):
    """Calculate and print evaluation metrics."""
    # Convert back from log scale
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate percentage error
    mape = np.mean(np.abs((y_true_exp - y_pred_exp) / y_true_exp)) * 100
    
    # Print metrics
    print(f"\n{model_name} - {set_name} Set:")
    print(f"  RMSE (log scale): {rmse:.4f}")
    print(f"  MAE (log scale): {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"\n  RMSE (original scale): ${np.expm1(rmse):,.2f}")
    
    return {
        'model': model_name,
        'set': set_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def plot_predictions(y_true, y_pred, model_name, set_name):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    
    # Convert back to original scale
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    
    # Create scatter plot
    plt.scatter(y_true_exp, y_pred_exp, alpha=0.5)
    
    # Add a diagonal line
    max_val = max(y_true_exp.max(), y_pred_exp.max()) * 1.05
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title(f'Actual vs Predicted House Prices - {model_name} ({set_name} Set)')
    
    # Save the plot
    plot_filename = f"prediction_plot_{model_name.lower().replace(' ', '_')}_{set_name.lower()}.png"
    plot_path = os.path.join(EVAL_DIR, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def plot_residuals(y_true, y_pred, model_name, set_name):
    """Plot residuals vs predicted values."""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values (log scale)')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'Residual Plot - {model_name} ({set_name} Set)')
    
    # Save the plot
    plot_filename = f"residual_plot_{model_name.lower().replace(' ', '_')}_{set_name.lower()}.png"
    plot_path = os.path.join(EVAL_DIR, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def load_data():
    """Load the preprocessed data."""
    print("Loading data...")
    
    # Load training data (for final evaluation on the full training set)
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    
    # Load test data (if available)
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    test_ids = pd.read_csv(os.path.join(DATA_DIR, 'test_ids.csv'))
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, y_train, X_test, test_ids

def load_models():
    """Load all trained models."""
    print("\nLoading models...")
    
    models = {}
    model_files = [
        'ridge.joblib',
        'ridge_tuned.joblib',
        'gradient_boosting.joblib',
        'gradient_boosting_tuned.joblib',
        'lightgbm.joblib',
        'lightgbm_tuned.joblib'
    ]
    
    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        if os.path.exists(model_path):
            model_name = model_file.replace('.joblib', '').replace('_', ' ').title()
            models[model_name] = joblib.load(model_path)
            print(f"  - Loaded {model_name}")
        else:
            print(f"  - Model not found: {model_file}")
    
    return models

def main():
    """Main function to evaluate models."""
    # Load data
    X_train, y_train, X_test, test_ids = load_data()
    
    # Load models
    models = load_models()
    
    if not models:
        print("No models found. Please train models first.")
        return
    
    results = []
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}...")
        
        # Make predictions on training set
        y_pred_train = model.predict(X_train)
        
        # Evaluate on training set
        train_metrics = evaluate_model(y_train, y_pred_train, model_name, "Training")
        results.append(train_metrics)
        
        # Generate plots for training set
        plot_predictions(y_train, y_pred_train, model_name, "Training")
        plot_residuals(y_train, y_pred_train, model_name, "Training")
        
        # If test set is available, evaluate on test set
        if X_test is not None:
            # For the competition, we don't have the true target for the test set
            # So we'll just generate the predictions for submission
            y_pred_test = model.predict(X_test)
            
            # Save predictions for submission
            if 'tuned' in model_name.lower():
                submission = pd.DataFrame({
                    'Id': test_ids['Id'],
                    'SalePrice': np.expm1(y_pred_test)  # Convert back to original scale
                })
                submission_file = os.path.join(EVAL_DIR, f'submission_{model_name.lower().replace(" ", "_")}.csv')
                submission.to_csv(submission_file, index=False)
                print(f"  - Submission file saved to {submission_file}")
    
    # Save results to a DataFrame
    results_df = pd.DataFrame(results)
    results_file = os.path.join(EVAL_DIR, 'model_evaluation_results.csv')
    results_df.to_csv(results_file, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("Model Evaluation Summary:")
    print("="*50)
    print(results_df[['model', 'set', 'rmse', 'r2', 'mape']].to_string(index=False))
    
    print(f"\nEvaluation complete! Results saved to {EVAL_DIR}")

if __name__ == "__main__":
    main()
