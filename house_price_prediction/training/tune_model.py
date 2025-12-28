import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from house_price_prediction.preprocessing.cleaner import DataCleaner
from house_price_prediction.preprocessing.encoder import CategoricalEncoder
from house_price_prediction.preprocessing.scaler import FeatureScaler
from house_price_prediction.features.engineer import FeatureEngineer
from house_price_prediction.logger import get_logger

logger = get_logger(__name__)

# Config
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'house_price_prediction', 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    return pd.read_csv(TRAIN_FILE)

def tune_random_forest():
    logger.info("Starting Hyperparameter Tuning for RandomForest...")
    
    # Load data
    df = load_data()
    X = df.drop(columns=['SalePrice', 'Id'], errors='ignore')
    y = df['SalePrice']
    y_log = np.log1p(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    # Preprocessing setup
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
    categorical_with_na = ['GarageType', 'GarageFinish', 'BsmtQual', 'BsmtCond']
    log_transform_features = ['LotFrontage', 'LotArea', 'GrLivArea']
    
    cleaner = DataCleaner(
        cols_to_drop=cols_to_drop,
        categorical_with_na=categorical_with_na,
        numerical_cols=numerical_cols,
        log_transform_features=log_transform_features
    )
    
    pipeline = Pipeline([
        ('cleaner', cleaner),
        ('engineer', FeatureEngineer()),
        ('encoder', CategoricalEncoder()),
        ('scaler', FeatureScaler()),
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    # Define param grid
    # Note: We need to prefix params with 'model__' because it's in a pipeline
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    
    logger.info("Running GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    logger.info(f"Best Parameters: {grid_search.best_params_}")
    
    # Evaluate
    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)
    
    logger.info(f"Tuned RandomForest - RMSE: {rmse:.2f}, R2: {r2:.4f}")
    
    # Save
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'RandomForest_tuned_pipeline.joblib'))
    logger.info("Saved tuned model.")

if __name__ == "__main__":
    tune_random_forest()
