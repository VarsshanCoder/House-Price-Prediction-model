from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.numerical_cols = []

    def fit(self, X, y=None):
        # Identify numerical columns (excluding binary encoded ones if any, though StandardScaler usually applies to all)
        # Here we assume all remaining columns after encoding are numerical
        self.numerical_cols = X.columns.tolist()
        if self.numerical_cols:
            self.scaler.fit(X[self.numerical_cols])
        return self

    def transform(self, X):
        X = X.copy()
        if self.numerical_cols:
            # Add missing columns with mean (0 since we use StandardScaler, but actually we should use the mean from scaler if possible, or just 0 if we assume mean-centered)
            # Better: fill with 0 (mean) if missing.
            for col in self.numerical_cols:
                if col not in X.columns:
                    X[col] = 0 # Assuming 0 is a safe default for scaled data? No, this is BEFORE scaling.
                    # If we put 0 before scaling, it will be scaled to (0 - mean) / std.
                    # We want the result to be 0 (mean).
                    # So we should put the mean value.
                    # But we don't have easy access to original means here without inspecting scaler.mean_
                    # Actually, let's just fill with 0 for now, or handle it better.
                    # If we fill with 0, it's a specific value.
                    # A better approach is to ensure app.py provides defaults.
                    pass
            
            # Check which columns are actually present
            present_cols = [col for col in self.numerical_cols if col in X.columns]
            missing_cols = [col for col in self.numerical_cols if col not in X.columns]
            
            if missing_cols:
                 # For missing columns, we can't easily scale them correctly without the mean.
                 # Let's fill them with the mean from the scaler if available.
                 if hasattr(self.scaler, 'mean_'):
                     for i, col in enumerate(self.numerical_cols):
                         if col in missing_cols:
                             X[col] = self.scaler.mean_[i]
                 else:
                     for col in missing_cols:
                         X[col] = 0

            scaled_data = self.scaler.transform(X[self.numerical_cols])
            X = pd.DataFrame(scaled_data, columns=self.numerical_cols, index=X.index)
        return X
