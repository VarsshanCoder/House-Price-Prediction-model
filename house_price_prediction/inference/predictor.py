import joblib
import os
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'models')

class HousePricePredictor:
    def __init__(self, model_name='Lasso'):
        self.model_name = model_name
        self.pipeline = None
        self.load_model()

    def load_model(self):
        pipeline_path = os.path.join(MODEL_DIR, f'{self.model_name}_pipeline.joblib')
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Model file not found at {pipeline_path}. Please train the model first.")
        self.pipeline = joblib.load(pipeline_path)

    def predict(self, data):
        """
        Predict house price.
        data: dict or DataFrame
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Predict log price
        log_price = self.pipeline.predict(data)
        
        # Convert back to original scale
        price = np.expm1(log_price)
        
        return price[0]

if __name__ == "__main__":
    # Test
    predictor = HousePricePredictor()
    sample_data = {
        'MSSubClass': 60,
        'MSZoning': 'RL',
        'LotArea': 8450,
        'Neighborhood': 'CollgCr',
        'YearBuilt': 2003,
        'YrSold': 2008,
        'GrLivArea': 1710,
        'TotalBsmtSF': 856,
        'FullBath': 2,
        'HalfBath': 1,
        'BsmtFullBath': 1,
        'BsmtHalfBath': 0,
        'GarageArea': 548,
        'GarageCars': 2,
        'OverallQual': 7
    }
    price = predictor.predict(sample_data)
    print(f"Predicted Price: ${price:,.2f}")
