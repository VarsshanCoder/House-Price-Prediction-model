"""
Exploratory Data Analysis (EDA) for House Price Prediction

This script performs an initial exploration of the House Prices dataset.
It includes data loading, summary statistics, and visualizations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# File paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')

# Create output directory for plots
os.makedirs('output/eda', exist_ok=True)

def load_data():
    """Load the training data."""
    print("Loading training data...")
    return pd.read_csv(TRAIN_FILE)

def basic_info(df):
    """Display basic information about the dataset."""
    print("\n=== Dataset Information ===")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    print("\n=== Data Types ===")
    print(df.dtypes.value_counts())
    
    print("\n=== First 5 rows ===")
    print(df.head())

def analyze_target(df, target_col='SalePrice'):
    """Analyze the target variable."""
    print("\n=== Target Variable Analysis ===")
    print(df[target_col].describe())
    
    # Plot distribution of the target variable
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[target_col], kde=True)
    plt.title('Distribution of Sale Prices')
    plt.xlabel('Sale Price ($)')
    
    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[target_col])
    plt.title('Boxplot of Sale Prices')
    plt.ylabel('Sale Price ($)')
    
    plt.tight_layout()
    plt.savefig('output/eda/target_distribution.png')
    plt.close()

def analyze_missing_values(df):
    """Analyze and visualize missing values in the dataset."""
    print("\n=== Missing Values Analysis ===")
    
    # Calculate missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    }).sort_values('Percentage', ascending=False)
    
    print("\nColumns with missing values:")
    print(missing_df)
    
    # Plot missing values
    if not missing_df.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=missing_df.index, y='Percentage', data=missing_df)
        plt.xticks(rotation=90)
        plt.title('Percentage of Missing Values by Column')
        plt.tight_layout()
        plt.savefig('output/eda/missing_values.png')
        plt.close()

def analyze_numerical_features(df, target_col='SalePrice'):
    """Analyze numerical features and their relationship with the target."""
    print("\n=== Numerical Features Analysis ===")
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove ID and target column
    numerical_cols = [col for col in numerical_cols if col not in ['Id', target_col]]
    
    # Plot correlation heatmap
    plt.figure(figsize=(16, 12))
    correlation_matrix = df[numerical_cols + [target_col]].corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=True, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('output/eda/correlation_heatmap.png')
    plt.close()
    
    # Get top 10 features most correlated with the target
    top_features = correlation_matrix[target_col].sort_values(ascending=False).index[1:11]
    
    # Plot relationships with top features
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[feature], y=df[target_col])
        plt.title(f'{feature} vs {target_col}')
        plt.xlabel(feature)
        plt.ylabel('Sale Price ($)')
        plt.tight_layout()
        plt.savefig(f'output/eda/{feature}_vs_{target_col}.png')
        plt.close()

def analyze_categorical_features(df, target_col='SalePrice', max_categories=10):
    """Analyze categorical features and their relationship with the target."""
    print("\n=== Categorical Features Analysis ===")
    
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        print("No categorical columns found.")
        return
    
    # For each categorical column, plot the distribution and relationship with target
    for col in categorical_cols[:max_categories]:  # Limit to first 10 for brevity
        # Count of each category
        plt.figure(figsize=(12, 6))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(f'output/eda/{col}_distribution.png')
        plt.close()
        
        # Relationship with target
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=col, y=target_col, data=df)
        plt.xticks(rotation=90)
        plt.title(f'{col} vs {target_col}')
        plt.tight_layout()
        plt.savefig(f'output/eda/{col}_vs_{target_col}.png')
        plt.close()

def main():
    """Main function to run the EDA."""
    # Load the data
    df = load_data()
    
    # Perform EDA
    basic_info(df)
    analyze_target(df)
    analyze_missing_values(df)
    analyze_numerical_features(df)
    analyze_categorical_features(df)
    
    print("\n=== EDA Complete! Check the 'output/eda' directory for visualizations. ===")

if __name__ == "__main__":
    main()
