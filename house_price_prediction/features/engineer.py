from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Property Age
        if 'YearBuilt' in X.columns and 'YrSold' in X.columns:
            X['PropertyAge'] = X['YrSold'] - X['YearBuilt']
            X['PropertyAge'] = X['PropertyAge'].apply(lambda x: max(0, x)) # Ensure no negative age
        
        # Total Square Footage
        if 'GrLivArea' in X.columns and 'TotalBsmtSF' in X.columns:
            X['TotalSqFt'] = X['GrLivArea'] + X['TotalBsmtSF']
        
        # Total Bathrooms
        if all(col in X.columns for col in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
            X['TotalBath'] = (X['FullBath'] + 
                              0.5 * X['HalfBath'] + 
                              X['BsmtFullBath'] + 
                              0.5 * X['BsmtHalfBath'])
        
        # Amenity Score
        # Simple count of luxury items
        X['AmenityScore'] = 0
        if 'Fireplaces' in X.columns:
            X['AmenityScore'] += (X['Fireplaces'] > 0).astype(int)
        if 'PoolArea' in X.columns:
            X['AmenityScore'] += (X['PoolArea'] > 0).astype(int)
        if 'GarageArea' in X.columns:
            X['AmenityScore'] += (X['GarageArea'] > 0).astype(int)
        if 'WoodDeckSF' in X.columns:
            X['AmenityScore'] += (X['WoodDeckSF'] > 0).astype(int)
        if 'OpenPorchSF' in X.columns:
            X['AmenityScore'] += (X['OpenPorchSF'] > 0).astype(int)
            
        return X
