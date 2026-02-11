"""
Models Module
Handles training, evaluation, and comparison of ML models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: Optional[Dict] = None,
    random_state: int = 42
) -> LogisticRegression:
    """
    Train Logistic Regression model (baseline, interpretable).
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weight: Optional class weights for imbalanced data
        random_state: Random seed
        
    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight or 'balanced',
        random_state=random_state,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 5,
    class_weight: Optional[Dict] = None,
    random_state: int = 42
) -> DecisionTreeClassifier:
    """
    Train Decision Tree model (interpretable, rule-based).
    
    Args:
        X_train: Training features
        y_train: Training labels
        max_depth: Maximum tree depth (controls interpretability)
        class_weight: Optional class weights
        random_state: Random seed
        
    Returns:
        Trained DecisionTreeClassifier
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight=class_weight or 'balanced',
        random_state=random_state,
        min_samples_split=10,
        min_samples_leaf=5
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 10,
    class_weight: Optional[Dict] = None,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train Random Forest model (ensemble, high performance).
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        class_weight: Optional class weights
        random_state: Random seed
        
    Returns:
        Trained RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight or 'balanced',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> GradientBoostingClassifier:
    """
    Train Gradient Boosting model (high performance).
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of boosting stages
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
        
    Returns:
        Trained GradientBoostingClassifier
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple models across metrics.
    
    Args:
        models: Dictionary of model_name -> trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        results.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc']
        })
    
    return pd.DataFrame(results).round(4)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Confusion Matrix',
    labels: list = ['Good Credit', 'Bad Credit'],
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix with annotations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        labels: Class labels
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    return fig


def plot_roc_curves(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Args:
        models: Dictionary of model_name -> trained model
        X_test: Test features
        y_test: Test labels
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for (name, model), color in zip(models.items(), colors):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Model Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 15,
    title: str = 'Feature Importance',
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance from tree-based models.
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importances[indices], color='steelblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    return fig


def get_business_threshold(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 5.0
) -> Tuple[float, Dict]:
    """
    Find optimal threshold based on business costs.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        false_positive_cost: Cost of wrongly rejecting good customer
        false_negative_cost: Cost of wrongly approving bad customer
        
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    best_threshold = 0.5
    min_cost = float('inf')
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        # Cost calculation
        fp = cm[0, 1]  # False positives (good customers rejected)
        fn = cm[1, 0]  # False negatives (bad customers approved)
        
        total_cost = fp * false_positive_cost + fn * false_negative_cost
        
        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = threshold
    
    # Final metrics at optimal threshold
    y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
    metrics = evaluate_model(model, X_test, y_test, threshold=best_threshold)
    metrics['threshold'] = best_threshold
    metrics['total_cost'] = min_cost
    
    return best_threshold, metrics
