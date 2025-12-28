from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from .registry import registry

def register_models():
    # Baseline
    registry.register('LinearRegression', LinearRegression())
    
    # Regularized Linear Models
    registry.register('Ridge', Ridge(alpha=1.0))
    registry.register('Lasso', Lasso(alpha=0.001)) # Small alpha for Lasso to avoid zeroing out too many
    
    # Tree-based Models
    registry.register('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42))
    registry.register('XGBoost', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))

register_models()
