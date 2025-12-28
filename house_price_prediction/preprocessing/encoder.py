from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, handle_unknown='ignore'):
        self.handle_unknown = handle_unknown
        self.encoder = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False)
        self.categorical_cols = []
        self.feature_names_out = []

    def fit(self, X, y=None):
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.categorical_cols:
            self.encoder.fit(X[self.categorical_cols])
            self.feature_names_out = self.encoder.get_feature_names_out(self.categorical_cols)
        return self

    def transform(self, X):
        X = X.copy()
        if self.categorical_cols:
            # Add missing columns with 'None' (or whatever was used for missing during fit)
            for col in self.categorical_cols:
                if col not in X.columns:
                    X[col] = 'None'
            
            encoded_data = self.encoder.transform(X[self.categorical_cols])
            encoded_df = pd.DataFrame(encoded_data, columns=self.feature_names_out, index=X.index)
            X = X.drop(columns=self.categorical_cols)
            X = pd.concat([X, encoded_df], axis=1)
        return X
