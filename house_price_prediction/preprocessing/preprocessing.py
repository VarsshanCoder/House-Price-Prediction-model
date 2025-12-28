"""
Data Preprocessing for House Price Prediction

This script handles the cleaning and preprocessing of the House Prices dataset.
It includes handling missing values, encoding categorical variables, and feature scaling.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# File paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'preprocessing')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
TARGET_COL = 'SalePrice'
ID_COL = 'Id'

# Columns to drop (based on EDA and domain knowledge)
# These columns have too many missing values or are not useful for prediction
COLS_TO_DROP = [
    'PoolQC',           # Too many missing values (99.5%)
    'MiscFeature',      # Too many missing values (96.3%)
    'Alley',            # Too many missing values (93.8%)
    'Fence',            # Too many missing values (80.8%)
    'FireplaceQu',      # Too many missing values (47.3%)
    'Id',               # Just an identifier, not useful for prediction
]

# Categorical columns with NA as a category (these are not missing, but represent 'None' or 'Not Available')
CATEGORICAL_WITH_NA = [
    'GarageType',
    'GarageFinish',
    'GarageQual',
    'GarageCond',
    'BsmtQual',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'MasVnrType',
]

# Numerical columns to impute
NUMERICAL_COLS = [
    'LotFrontage',
    'MasVnrArea',
    'GarageYrBlt',
]

# Features to log transform (right-skewed features from EDA)
LOG_TRANSFORM_FEATURES = [
    'LotFrontage',
    'LotArea',
    'MasVnrArea',
    'BsmtFinSF1',
    'BsmtFinSF2',
    'TotalBsmtSF',
    '1stFlrSF',
    '2ndFlrSF',
    'GrLivArea',
    'GarageArea',
    'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch',
    '3SsnPorch',
    'ScreenPorch',
    'PoolArea',
    'MiscVal',
]

def load_data():
    """Load the training and test datasets."""
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """Preprocess the data."""
    print("\nPreprocessing data...")
    
    # Save the target variable and ID
    y_train = train_df[TARGET_COL]
    test_ids = test_df[ID_COL]
    
    # Drop the target and ID columns from the features
    train_df = train_df.drop(columns=[TARGET_COL, ID_COL], errors='ignore')
    test_df = test_df.drop(columns=[ID_COL], errors='ignore')
    
    # Drop columns with too many missing values or not useful for prediction
    print(f"Dropping columns: {COLS_TO_DROP}")
    train_df = train_df.drop(columns=COLS_TO_DROP, errors='ignore')
    test_df = test_df.drop(columns=COLS_TO_DROP, errors='ignore')
    
    # Handle missing values in categorical columns
    for col in CATEGORICAL_WITH_NA:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna('None')
            test_df[col] = test_df[col].fillna('None')
    
    # Handle missing values in numerical columns
    for col in NUMERICAL_COLS:
        if col in train_df.columns:
            # For LotFrontage, use the median of the neighborhood
            if col == 'LotFrontage':
                train_df[col] = train_df.groupby('Neighborhood')[col].transform(
                    lambda x: x.fillna(x.median()))
                test_df[col] = test_df.groupby('Neighborhood')[col].transform(
                    lambda x: x.fillna(x.median()))
            # For other numerical columns, use the median
            else:
                median_val = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_val)
                test_df[col] = test_df[col].fillna(median_val)
    
    # Log transform right-skewed features
    for col in LOG_TRANSFORM_FEATURES:
        if col in train_df.columns:
            # Add 1 to handle zeros
            train_df[col] = np.log1p(train_df[col])
            test_df[col] = np.log1p(test_df[col])
    
    # Log transform the target variable (SalePrice)
    y_train = np.log1p(y_train)
    
    return train_df, test_df, y_train, test_ids

def create_preprocessing_pipeline():
    """Create a preprocessing pipeline for the data."""
    print("\nCreating preprocessing pipeline...")
    
    # Identify categorical and numerical columns
    categorical_cols = []
    numerical_cols = []
    
    # These are the columns that should be treated as categorical
    # We'll get the actual columns that exist in our data
    potential_categorical = [
        'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
        'SaleType', 'SaleCondition'
    ]
    
    # We'll determine the actual columns in the data when we fit the pipeline
    
    # Numerical transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformers
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # Drop columns that we don't explicitly list
    )
    
    return preprocessor

def save_preprocessed_data(X_train, X_test, y_train, test_ids, preprocessor, output_dir=OUTPUT_DIR):
    """Save the preprocessed data and the preprocessor."""
    print("\nSaving preprocessed data...")
    
    # Save the preprocessed data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    
    # Save the test IDs for later use in submission
    test_ids.to_csv(os.path.join(output_dir, 'test_ids.csv'), index=False)
    
    # Save the preprocessor for later use in inference
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
    
    print(f"Preprocessed data saved to {output_dir}")

def main():
    """Main function to run the preprocessing pipeline."""
    # Load the data
    train_df, test_df = load_data()
    
    # Preprocess the data
    train_df_processed, test_df_processed, y_train, test_ids = preprocess_data(train_df, test_df)
    
    # Create and fit the preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Identify categorical and numerical columns
    categorical_cols = train_df_processed.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = train_df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Update the column transformer with the actual columns
    preprocessor.named_transformers_['cat'].set_params(onehot__categories='auto')
    preprocessor.set_params(
        num__imputer__strategy='median',
        num__scaler=StandardScaler(),
        cat__onehot__handle_unknown='ignore',
        cat__onehot__sparse_output=False
    )
    
    # Update the column transformer with the actual columns
    preprocessor.transformers = [
        ('num', preprocessor.named_transformers_['num'], numerical_cols),
        ('cat', preprocessor.named_transformers_['cat'], categorical_cols)
    ]
    
    # Fit and transform the training data
    print("\nFitting and transforming the training data...")
    X_train = preprocessor.fit_transform(train_df_processed)
    
    # Transform the test data
    print("Transforming the test data...")
    X_test = preprocessor.transform(test_df_processed)
    
    # Save the preprocessed data and the preprocessor
    save_preprocessed_data(X_train, X_test, y_train, test_ids, preprocessor)
    
    print("\n=== Preprocessing complete! ===")
    print(f"Training data shape after preprocessing: {X_train.shape}")
    print(f"Test data shape after preprocessing: {X_test.shape}")

if __name__ == "__main__":
    main()
