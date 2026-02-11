"""
Fairness Module
Bias analysis and fairness metrics for model auditing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    sensitive_attr: np.ndarray,
    group_names: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Calculate performance metrics by sensitive group.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        sensitive_attr: Sensitive attribute values
        group_names: Optional mapping of values to names
        
    Returns:
        DataFrame with metrics by group
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score
    )
    
    groups = np.unique(sensitive_attr)
    results = []
    
    for group in groups:
        mask = sensitive_attr == group
        group_name = group_names.get(group, str(group)) if group_names else str(group)
        
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        y_proba_g = y_pred_proba[mask]
        
        # Calculate metrics
        n_samples = len(y_true_g)
        n_positive = y_pred_g.sum()
        positive_rate = n_positive / n_samples if n_samples > 0 else 0
        
        metrics = {
            'group': group_name,
            'n_samples': n_samples,
            'n_positive_pred': n_positive,
            'positive_rate': positive_rate,
            'accuracy': accuracy_score(y_true_g, y_pred_g),
            'precision': precision_score(y_true_g, y_pred_g, zero_division=0),
            'recall': recall_score(y_true_g, y_pred_g, zero_division=0),
            'f1_score': f1_score(y_true_g, y_pred_g, zero_division=0)
        }
        
        # ROC-AUC (only if both classes present)
        if len(np.unique(y_true_g)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true_g, y_proba_g)
        else:
            metrics['roc_auc'] = np.nan
        
        # True Positive Rate (TPR) and False Positive Rate (FPR)
        n_actual_pos = y_true_g.sum()
        n_actual_neg = len(y_true_g) - n_actual_pos
        
        tp = ((y_pred_g == 1) & (y_true_g == 1)).sum()
        fp = ((y_pred_g == 1) & (y_true_g == 0)).sum()
        
        metrics['tpr'] = tp / n_actual_pos if n_actual_pos > 0 else 0
        metrics['fpr'] = fp / n_actual_neg if n_actual_neg > 0 else 0
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def calculate_disparate_impact(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    privileged_group: str,
    unprivileged_group: str
) -> Dict[str, float]:
    """
    Calculate disparate impact ratio (80% rule).
    
    A ratio below 0.8 or above 1.25 may indicate discrimination.
    
    Args:
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values
        privileged_group: Value for privileged group
        unprivileged_group: Value for unprivileged group
        
    Returns:
        Dictionary with disparate impact metrics
    """
    # Positive prediction rates
    priv_mask = sensitive_attr == privileged_group
    unpriv_mask = sensitive_attr == unprivileged_group
    
    priv_rate = y_pred[priv_mask].mean()
    unpriv_rate = y_pred[unpriv_mask].mean()
    
    # Disparate impact ratio
    if priv_rate > 0:
        di_ratio = unpriv_rate / priv_rate
    else:
        di_ratio = np.nan
    
    # 80% rule check
    passes_80_rule = 0.8 <= di_ratio <= 1.25
    
    return {
        'privileged_rate': priv_rate,
        'unprivileged_rate': unpriv_rate,
        'disparate_impact_ratio': di_ratio,
        'passes_80_rule': passes_80_rule,
        'bias_direction': 'against_unprivileged' if di_ratio < 0.8 else 
                          ('against_privileged' if di_ratio > 1.25 else 'neutral')
    }


def calculate_equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    group_a: str,
    group_b: str
) -> Dict[str, float]:
    """
    Calculate equalized odds (difference in TPR and FPR between groups).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values
        group_a: First group value
        group_b: Second group value
        
    Returns:
        Dictionary with equalized odds metrics
    """
    def calculate_rates(mask):
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        n_pos = y_t.sum()
        n_neg = len(y_t) - n_pos
        
        tp = ((y_p == 1) & (y_t == 1)).sum()
        fp = ((y_p == 1) & (y_t == 0)).sum()
        
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        
        return tpr, fpr
    
    mask_a = sensitive_attr == group_a
    mask_b = sensitive_attr == group_b
    
    tpr_a, fpr_a = calculate_rates(mask_a)
    tpr_b, fpr_b = calculate_rates(mask_b)
    
    return {
        f'{group_a}_tpr': tpr_a,
        f'{group_a}_fpr': fpr_a,
        f'{group_b}_tpr': tpr_b,
        f'{group_b}_fpr': fpr_b,
        'tpr_difference': abs(tpr_a - tpr_b),
        'fpr_difference': abs(fpr_a - fpr_b),
        'equalized_odds_satisfied': (abs(tpr_a - tpr_b) < 0.1) and (abs(fpr_a - fpr_b) < 0.1)
    }


def generate_bias_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    sensitive_features: Dict[str, np.ndarray],
    group_mappings: Optional[Dict[str, Dict]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive bias analysis report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        sensitive_features: Dict of feature_name -> values
        group_mappings: Optional mappings for group names
        
    Returns:
        Dictionary with complete bias analysis
    """
    report = {
        'overall_metrics': {},
        'group_metrics': {},
        'disparate_impact': {},
        'equalized_odds': {},
        'recommendations': []
    }
    
    # Overall metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    report['overall_metrics'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'positive_rate': y_pred.mean()
    }
    
    # Analyze each sensitive feature
    for feature_name, values in sensitive_features.items():
        group_names = group_mappings.get(feature_name, {}) if group_mappings else {}
        
        # Group metrics
        metrics_df = calculate_group_metrics(
            y_true, y_pred, y_pred_proba, values, group_names
        )
        report['group_metrics'][feature_name] = metrics_df
        
        # Disparate impact for binary features
        unique_values = np.unique(values)
        if len(unique_values) == 2:
            di = calculate_disparate_impact(
                y_pred, values, unique_values[0], unique_values[1]
            )
            report['disparate_impact'][feature_name] = di
            
            # Equalized odds
            eo = calculate_equalized_odds(
                y_true, y_pred, values, unique_values[0], unique_values[1]
            )
            report['equalized_odds'][feature_name] = eo
            
            # Generate recommendations
            if not di['passes_80_rule']:
                report['recommendations'].append(
                    f"WARNING - {feature_name}: Disparate impact ratio = {di['disparate_impact_ratio']:.2f}. "
                    f"Consider reviewing model for potential bias."
                )
            
            if not eo['equalized_odds_satisfied']:
                if eo['tpr_difference'] > 0.1:
                    report['recommendations'].append(
                        f"WARNING - {feature_name}: TPR difference = {eo['tpr_difference']:.2f}. "
                        f"Groups have different true positive rates."
                    )
                if eo['fpr_difference'] > 0.1:
                    report['recommendations'].append(
                        f"WARNING - {feature_name}: FPR difference = {eo['fpr_difference']:.2f}. "
                        f"Groups have different false positive rates."
                    )
    
    if not report['recommendations']:
        report['recommendations'].append(
            "PASS - No significant bias detected across analyzed sensitive features."
        )
    
    return report


def plot_fairness_comparison(
    metrics_df: pd.DataFrame,
    metric: str = 'positive_rate',
    title: str = 'Fairness Comparison',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot metric comparison across groups.
    
    Args:
        metrics_df: DataFrame from calculate_group_metrics
        metric: Metric to compare
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    groups = metrics_df['group'].tolist()
    values = metrics_df[metric].tolist()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    bars = ax.bar(groups, values, color=colors, edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2%}', ha='center', va='bottom', fontsize=10
        )
    
    # Add 80% rule reference lines if showing positive rate
    if metric == 'positive_rate':
        mean_rate = np.mean(values)
        ax.axhline(y=mean_rate * 0.8, color='red', linestyle='--', 
                   alpha=0.7, label='80% threshold')
        ax.axhline(y=mean_rate * 1.25, color='red', linestyle='--', 
                   alpha=0.7, label='125% threshold')
    
    ax.set_xlabel('Group')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_group_comparison_radar(
    metrics_df: pd.DataFrame,
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Create radar chart comparing groups across metrics.
    
    Args:
        metrics_df: DataFrame from calculate_group_metrics
        metrics: List of metrics to compare
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    groups = metrics_df['group'].tolist()
    
    # Number of metrics
    N = len(metrics)
    
    # Compute angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    
    for i, group in enumerate(groups):
        values = metrics_df[metrics_df['group'] == group][metrics].values.flatten().tolist()
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=group)
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Performance Metrics by Group', pad=20)
    
    plt.tight_layout()
    return fig


def create_age_bins(
    age_values: np.ndarray,
    bins: List[int] = [0, 25, 35, 45, 55, 100],
    labels: List[str] = ['18-25', '26-35', '36-45', '46-55', '55+']
) -> np.ndarray:
    """
    Bin age values for fairness analysis.
    
    Args:
        age_values: Age values
        bins: Bin edges
        labels: Bin labels
        
    Returns:
        Binned age values
    """
    result = pd.cut(age_values, bins=bins, labels=labels).astype(str)
    # If already a numpy array, don't call .values
    if hasattr(result, 'values'):
        return result.values
    return np.array(result)


def create_gender_from_status(personal_status: np.ndarray) -> np.ndarray:
    """
    Extract gender from personal_status_sex column.
    
    South German Credit encoding:
    - 1: male divorced/separated
    - 2: female divorced/separated/married
    - 3: male single
    - 4: male married/widowed
    
    Args:
        personal_status: Personal status values
        
    Returns:
        Gender array ('male' or 'female')
    """
    return np.where(personal_status == 2, 'female', 'male')


def format_bias_report_markdown(report: Dict) -> str:
    """
    Format bias report as markdown for notebook display.
    
    Args:
        report: Report from generate_bias_report
        
    Returns:
        Markdown formatted string
    """
    lines = []
    
    lines.append("## Fairness & Bias Analysis Report\n")
    
    # Overall metrics
    lines.append("### Overall Model Performance")
    om = report['overall_metrics']
    lines.append(f"- Accuracy: {om['accuracy']:.2%}")
    lines.append(f"- Precision: {om['precision']:.2%}")
    lines.append(f"- Recall: {om['recall']:.2%}")
    lines.append(f"- Overall Positive Rate: {om['positive_rate']:.2%}\n")
    
    # Disparate impact
    if report['disparate_impact']:
        lines.append("### Disparate Impact Analysis")
        for feature, di in report['disparate_impact'].items():
            status = "PASS" if di['passes_80_rule'] else "FAIL"
            lines.append(f"\n**{feature}**")
            lines.append(f"- Disparate Impact Ratio: {di['disparate_impact_ratio']:.3f} {status}")
            lines.append(f"- Privileged Group Rate: {di['privileged_rate']:.2%}")
            lines.append(f"- Unprivileged Group Rate: {di['unprivileged_rate']:.2%}")
    
    # Recommendations
    lines.append("\n### Recommendations")
    for rec in report['recommendations']:
        lines.append(f"- {rec}")
    
    return "\n".join(lines)
