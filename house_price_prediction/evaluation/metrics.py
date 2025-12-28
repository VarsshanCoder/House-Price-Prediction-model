import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate RMSE, MAE, and R2 score.
    Assumes y_true and y_pred are in the original scale (not log-transformed).
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def print_metrics(metrics):
    print("=== Model Performance ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
