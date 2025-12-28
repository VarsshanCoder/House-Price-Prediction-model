import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from house_price_prediction.preprocessing.cleaner import DataCleaner
from house_price_prediction.preprocessing.encoder import CategoricalEncoder
from house_price_prediction.preprocessing.scaler import FeatureScaler
from house_price_prediction.features.engineer import FeatureEngineer
from house_price_prediction.models.registry import registry
import house_price_prediction.models.regressors # This triggers registration
from house_price_prediction.logger import get_logger

logger = get_logger(__name__)

# Config
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'house_price_prediction', 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    return pd.read_csv(TRAIN_FILE)

def train_all_models():
    logger.info("Training all models...")
    
    # Load data
    df = load_data()
    
    # Separate target
    X = df.drop(columns=['SalePrice', 'Id'], errors='ignore')
    y = df['SalePrice']
    y_log = np.log1p(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    y_test_orig = np.expm1(y_test)
    
    # Define preprocessing steps (same as before)
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
    categorical_with_na = ['GarageType', 'GarageFinish', 'BsmtQual', 'BsmtCond']
    log_transform_features = ['LotFrontage', 'LotArea', 'GrLivArea']
    
    cleaner = DataCleaner(
        cols_to_drop=cols_to_drop,
        categorical_with_na=categorical_with_na,
        numerical_cols=numerical_cols,
        log_transform_features=log_transform_features
    )
    
    engineer = FeatureEngineer()
    encoder = CategoricalEncoder()
    scaler = FeatureScaler()
    
    results = []
    
    for model_name in registry.list_models():
        logger.info(f"Training {model_name}...")
        model = registry.get_model(model_name)
        
        pipeline = Pipeline([
            ('cleaner', cleaner),
            ('engineer', engineer),
            ('encoder', encoder),
            ('scaler', scaler),
            ('model', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred_log = pipeline.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
        mae = mean_absolute_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)
        
        logger.info(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        results.append({
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f'{model_name}_pipeline.joblib')
        joblib.dump(pipeline, model_path)
    
    # Save comparison
    results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
    results_df.to_csv(os.path.join(MODEL_DIR, 'model_comparison.csv'), index=False)
    logger.info("=== Model Comparison ===")
    logger.info(f"\n{results_df}")

if __name__ == "__main__":
    train_all_models()
