"""
Preprocessing Module
Handles feature engineering, encoding, and train/test splitting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def create_age_groups(age_series: pd.Series) -> pd.Series:
    """
    Create age groups for fairness analysis.
    
    Args:
        age_series: Series containing age values
        
    Returns:
        Series with age group labels
    """
    bins = [0, 25, 35, 45, 55, 100]
    labels = ['18-25', '26-35', '36-45', '46-55', '55+']
    return pd.cut(age_series, bins=bins, labels=labels)


def extract_gender(personal_status_sex: pd.Series) -> pd.Series:
    """
    Extract gender from personal_status_sex column.
    
    In the South German Credit dataset:
    - 1: male divorced/separated
    - 2: female divorced/separated/married
    - 3: male single
    - 4: male married/widowed
    
    Args:
        personal_status_sex: Series with personal status/sex codes
        
    Returns:
        Series with 'male' or 'female' labels
    """
    return personal_status_sex.apply(lambda x: 'female' if x == 2 else 'male')


def prepare_data_for_modeling(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True
) -> Dict:
    """
    Prepare data for ML modeling with proper train/test split.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        scale_features: Whether to standardize numerical features
        
    Returns:
        Dictionary with train/test data and preprocessor
    """
    # Identify feature types
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Split data first (before any preprocessing to prevent leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class balance
    )
    
    # Create preprocessing pipeline
    if scale_features:
        numerical_transformer = StandardScaler()
    else:
        numerical_transformer = 'passthrough'
    
    if categorical_cols:
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols)
            ],
            remainder='passthrough'
        )
    
    # Fit on training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    feature_names = numerical_cols.copy()
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_.get('cat')
        if hasattr(cat_encoder, 'get_feature_names_out'):
            cat_features = cat_encoder.get_feature_names_out(categorical_cols).tolist()
            feature_names.extend(cat_features)
    
    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'X_train_df': X_train,
        'X_test_df': X_test,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'train_indices': X_train.index.tolist(),
        'test_indices': X_test.index.tolist()
    }


def prepare_data_simple(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Simple data preparation without complex preprocessing.
    Better for tree-based models and explainability.
    
    Args:
        X: Feature DataFrame (already numeric)
        y: Target Series
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with train/test data
    """
    # Ensure all columns are numeric
    X_numeric = X.copy()
    for col in X_numeric.columns:
        if X_numeric[col].dtype == 'object':
            X_numeric[col] = LabelEncoder().fit_transform(X_numeric[col])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X_numeric.columns.tolist(),
        'train_indices': X_train.index.tolist(),
        'test_indices': X_test.index.tolist()
    }


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that may improve model performance.
    
    Args:
        X: Original feature DataFrame
        
    Returns:
        DataFrame with additional derived features
    """
    X_new = X.copy()
    
    # Credit amount per month ratio
    if 'credit_amount' in X.columns and 'duration_months' in X.columns:
        X_new['monthly_payment'] = X['credit_amount'] / X['duration_months']
    
    # Age group
    if 'age' in X.columns:
        X_new['age_group'] = create_age_groups(X['age'])
    
    return X_new


def get_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target Series
        
    Returns:
        Dictionary mapping class to weight
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))
