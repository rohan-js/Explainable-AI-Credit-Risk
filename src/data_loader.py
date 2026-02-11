"""
Data Loader Module
Handles loading and initial parsing of the South German Credit dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


# Feature name mapping for readability
FEATURE_NAMES = {
    'laufkont': 'checking_account_status',
    'laufzeit': 'duration_months',
    'moral': 'credit_history',
    'verw': 'purpose',
    'hoession': 'credit_amount',
    'sparkont': 'savings_account',
    'besession': 'employment_duration',
    'rate': 'installment_rate_percent',
    'famgeschl': 'personal_status_sex',
    'bession': 'other_debtors_guarantors',
    'wession': 'present_residence_since',
    'vession': 'property',
    'alter': 'age',
    'weitession': 'other_installment_plans',
    'wession': 'housing',
    'bishession': 'num_existing_credits',
    'beruf': 'job',
    'untession': 'num_dependents',
    'telession': 'telephone',
    'gasession': 'foreign_worker',
    'kreession': 'credit_risk'
}

# Readable feature descriptions for explainability
FEATURE_DESCRIPTIONS = {
    'checking_account_status': 'Status of existing checking account',
    'duration_months': 'Duration of credit in months',
    'credit_history': 'Credit history with the bank',
    'purpose': 'Purpose of the loan',
    'credit_amount': 'Credit amount in DM',
    'savings_account': 'Savings account/bonds balance',
    'employment_duration': 'Present employment duration',
    'installment_rate_percent': 'Installment rate as % of disposable income',
    'personal_status_sex': 'Personal status and sex',
    'other_debtors_guarantors': 'Other debtors or guarantors',
    'present_residence_since': 'Years at present residence',
    'property': 'Property ownership',
    'age': 'Age in years',
    'other_installment_plans': 'Other installment plans',
    'housing': 'Housing situation',
    'num_existing_credits': 'Number of existing credits at this bank',
    'job': 'Job type',
    'num_dependents': 'Number of dependents',
    'telephone': 'Has telephone registered',
    'foreign_worker': 'Is foreign worker'
}

# Sensitive attributes for fairness analysis
SENSITIVE_ATTRIBUTES = ['age', 'personal_status_sex', 'foreign_worker']


def load_dataset_from_uci() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load the German Credit dataset.
    Priority: 1) Local CSV, 2) UCI API, 3) Direct URL download
    
    Returns:
        X: Feature DataFrame
        y: Target Series (1 = Bad credit/default, 0 = Good credit)
        metadata: Dataset metadata dictionary
    """
    import os
    
    # First try local CSV (from download_data.py)
    local_paths = ['data/german_credit.csv', '../data/german_credit.csv']
    for local_path in local_paths:
        if os.path.exists(local_path):
            print(f"Loading from local file: {local_path}")
            df = pd.read_csv(local_path)
            X = df.drop(columns=['credit_risk'])
            y = df['credit_risk']
            source = 'Local CSV'
            break
    else:
        # Try UCI API
        try:
            from ucimlrepo import fetch_ucirepo
            south_german_credit = fetch_ucirepo(id=573)
            X = south_german_credit.data.features.copy()
            y = south_german_credit.data.targets.copy()
            source = 'UCI ML Repository (API)'
        except Exception:
            # Fallback: Download directly from OpenML mirror
            print("UCI API unavailable, downloading from OpenML mirror...")
            url = "https://www.openml.org/data/get_csv/1586225/php0iVrYT"
            try:
                df = pd.read_csv(url)
            except Exception:
                # Second fallback: use German Credit from Statlog
                print("OpenML unavailable, using Statlog German Credit...")
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
                cols = ['checking_account_status', 'duration_months', 'credit_history', 'purpose',
                        'credit_amount', 'savings_account', 'employment_duration', 'installment_rate_percent',
                        'personal_status_sex', 'other_debtors_guarantors', 'present_residence_since',
                        'property', 'age', 'other_installment_plans', 'housing', 'num_existing_credits',
                        'job', 'num_dependents', 'telephone', 'foreign_worker', 'credit_risk']
                df = pd.read_csv(url, sep=' ', header=None, names=cols)
            
            # Separate features and target
            if 'credit_risk' in df.columns:
                X = df.drop(columns=['credit_risk'])
                y = df['credit_risk']
            elif 'kreession' in df.columns:
                X = df.drop(columns=['kreession'])
                y = df['kreession']
            else:
                # Assume last column is target
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
            source = 'OpenML/UCI Direct Download'
    
    # Flatten y if it's a DataFrame
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    # Convert target: 1 = Good (original), 2 = Bad (original)
    # Remap to: 1 = Bad (default), 0 = Good (no default)
    if y.max() == 2:
        y = (y == 2).astype(int)
    y.name = 'credit_risk'
    
    metadata = {
        'source': source,
        'dataset_id': 573,
        'name': 'South German Credit',
        'n_samples': len(X),
        'n_features': X.shape[1],
        'target_distribution': y.value_counts().to_dict()
    }
    
    return X, y, metadata


def load_dataset_from_csv(filepath: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load dataset from a local CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        X: Feature DataFrame
        y: Target Series
        metadata: Dataset metadata dictionary
    """
    df = pd.read_csv(filepath)
    
    # Assume last column is target
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Convert target if needed (1 = Good, 2 = Bad -> 0 = Good, 1 = Bad)
    if y.max() == 2:
        y = (y == 2).astype(int)
    
    y.name = 'credit_risk'
    
    metadata = {
        'source': 'Local CSV',
        'filepath': filepath,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'target_distribution': y.value_counts().to_dict()
    }
    
    return X, y, metadata


def get_feature_info(X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of all features for documentation.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        DataFrame with feature information
    """
    info = []
    for col in X.columns:
        desc = FEATURE_DESCRIPTIONS.get(col, 'No description available')
        is_sensitive = col in SENSITIVE_ATTRIBUTES
        
        info.append({
            'feature': col,
            'description': desc,
            'dtype': str(X[col].dtype),
            'n_unique': X[col].nunique(),
            'missing_pct': (X[col].isna().sum() / len(X)) * 100,
            'is_sensitive': is_sensitive
        })
    
    return pd.DataFrame(info)


def categorize_features(X: pd.DataFrame) -> Dict[str, list]:
    """
    Categorize features into numerical and categorical.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Dictionary with 'numerical' and 'categorical' feature lists
    """
    numerical = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Further split numerical into continuous and discrete
    continuous = [col for col in numerical if X[col].nunique() > 10]
    discrete = [col for col in numerical if X[col].nunique() <= 10]
    
    return {
        'numerical': numerical,
        'categorical': categorical,
        'continuous': continuous,
        'discrete': discrete,
        'sensitive': [col for col in X.columns if col in SENSITIVE_ATTRIBUTES]
    }
