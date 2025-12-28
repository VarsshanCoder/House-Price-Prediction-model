import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop=None, categorical_with_na=None, numerical_cols=None, log_transform_features=None):
        self.cols_to_drop = cols_to_drop if cols_to_drop else []
        self.categorical_with_na = categorical_with_na if categorical_with_na else []
        self.numerical_cols = numerical_cols if numerical_cols else []
        self.log_transform_features = log_transform_features if log_transform_features else []
        self.neighborhood_medians = {}
        self.global_medians = {}

    def fit(self, X, y=None):
        # Calculate medians for LotFrontage by Neighborhood
        if 'LotFrontage' in X.columns and 'Neighborhood' in X.columns:
            self.neighborhood_medians = X.groupby('Neighborhood')['LotFrontage'].median().to_dict()
        
        # Calculate global medians for other numerical columns
        for col in self.numerical_cols:
            if col in X.columns and col != 'LotFrontage':
                self.global_medians[col] = X[col].median()
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # Drop columns
        X = X.drop(columns=self.cols_to_drop, errors='ignore')
        
        # Handle missing values in categorical columns
        for col in self.categorical_with_na:
            if col in X.columns:
                X[col] = X[col].fillna('None')
        
        # Handle missing values in numerical columns
        for col in self.numerical_cols:
            if col in X.columns:
                if col == 'LotFrontage' and 'Neighborhood' in X.columns:
                    # Fill with neighborhood median, fallback to global median if neighborhood not found
                    X[col] = X.apply(
                        lambda row: self.neighborhood_medians.get(row['Neighborhood'], X['LotFrontage'].median())
                        if pd.isna(row[col]) else row[col],
                        axis=1
                    )
                elif col in self.global_medians:
                    X[col] = X[col].fillna(self.global_medians[col])
        
        # Log transform
        for col in self.log_transform_features:
            if col in X.columns:
                X[col] = np.log1p(X[col])
                
        return X
