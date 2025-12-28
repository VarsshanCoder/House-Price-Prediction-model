import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from house_price_prediction.training.train_pipeline import load_data, MODEL_DIR

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'explainability')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_shap_analysis(model_name='XGBoost'):
    print(f"Running SHAP analysis for {model_name}...")
    
    # Load pipeline
    pipeline_path = os.path.join(MODEL_DIR, f'{model_name}_pipeline.joblib')
    pipeline = joblib.load(pipeline_path)
    
    # Extract model and preprocessor
    # Assuming the last step is the model
    model = pipeline.named_steps['model']
    preprocessor = pipeline[:-1] # All steps except the last
    
    # Load data
    df = load_data()
    X = df.drop(columns=['SalePrice', 'Id'], errors='ignore')
    
    # Transform data
    # We use a subset for speed if dataset is large, but here it's small enough
    X_transformed = preprocessor.transform(X)
    
    # Get feature names
    # This is tricky with pipelines. 
    # My CategoricalEncoder and FeatureScaler return DataFrames with columns, so X_transformed should be a DataFrame
    # Let's verify if X_transformed is a DataFrame
    if isinstance(X_transformed, pd.DataFrame):
        feature_names = X_transformed.columns.tolist()
    else:
        # Fallback if it's numpy array (shouldn't be with my implementation)
        feature_names = [f'Feature {i}' for i in range(X_transformed.shape[1])]
    
    # Create explainer
    if model_name in ['LinearRegression', 'Ridge', 'Lasso']:
        # For linear models, we use LinearExplainer
        # We need to pass the masker (X_transformed) to the explainer for proper scaling
        explainer = shap.LinearExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
    else:
        # TreeExplainer for tree models (XGBoost, RandomForest)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
    
    # Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_shap_summary.png'))
    plt.close()
    
    # Force Plot for the first instance
    # shap.force_plot is interactive JS, we can save it as HTML
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], X_transformed.iloc[0,:], feature_names=feature_names, show=False)
    shap.save_html(os.path.join(OUTPUT_DIR, f'{model_name}_shap_force_plot.html'), force_plot)
    
    print(f"SHAP analysis saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_shap_analysis('Lasso')
