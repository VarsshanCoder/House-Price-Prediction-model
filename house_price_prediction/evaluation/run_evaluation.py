import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split

from house_price_prediction.training.train_pipeline import load_data, MODEL_DIR
from house_price_prediction.evaluation.visualizer import plot_actual_vs_predicted, plot_residuals

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'evaluation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_price_by_neighborhood(df):
    """
    Plot average price by neighborhood.
    This serves as a proxy for map-based visualization since we don't have lat/lon coordinates.
    """
    plt.figure(figsize=(12, 8))
    neighborhood_prices = df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
    sns.barplot(x=neighborhood_prices.values, y=neighborhood_prices.index, palette='viridis')
    plt.title('Average House Price by Neighborhood')
    plt.xlabel('Average Price ($)')
    plt.ylabel('Neighborhood')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'price_by_neighborhood.png'))
    plt.close()
    print(f"Neighborhood price plot saved to {OUTPUT_DIR}")

def run_evaluation(model_name='XGBoost'):
    print(f"Running evaluation for {model_name}...")
    
    # Load data
    df = load_data()
    
    # Generate Neighborhood plot (Spatial/Location insight)
    plot_price_by_neighborhood(df)
    
    # Prepare data for model evaluation
    X = df.drop(columns=['SalePrice', 'Id'], errors='ignore')
    y = df['SalePrice']
    y_log = np.log1p(y)
    
    # Split data (same random_state as training to ensure consistency)
    _, X_test, _, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    # Load model
    pipeline_path = os.path.join(MODEL_DIR, f'{model_name}_pipeline.joblib')
    if not os.path.exists(pipeline_path):
        print(f"Model {model_name} not found. Please train it first.")
        return
        
    pipeline = joblib.load(pipeline_path)
    
    # Predict
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    
    # Generate plots
    plot_actual_vs_predicted(y_test_orig, y_pred, model_name)
    plot_residuals(y_test_orig, y_pred, model_name)
    
    print(f"Evaluation plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_evaluation('Lasso')
