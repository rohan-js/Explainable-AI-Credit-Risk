"""
Explainers Module
SHAP and LIME implementations for global and local explainability.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
import warnings

# Suppress SHAP/LIME warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class GlobalExplainer:
    """
    Handles global model explainability using SHAP.
    """
    
    def __init__(self, model, X_train: np.ndarray, feature_names: List[str]):
        """
        Initialize the explainer.
        
        Args:
            model: Trained classifier
            X_train: Training data for background distribution
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.shap_values = None
        self.explainer = None
        
        # Choose appropriate explainer based on model type
        self._create_explainer()
    
    def _create_explainer(self):
        """Create the appropriate SHAP explainer based on model type."""
        model_name = type(self.model).__name__
        
        # Sample background data for efficiency
        if len(self.X_train) > 100:
            background = shap.sample(self.X_train, 100)
        else:
            background = self.X_train
        
        if model_name in ['RandomForestClassifier', 'GradientBoostingClassifier', 
                           'DecisionTreeClassifier']:
            # Use TreeExplainer for tree-based models (faster)
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Use KernelExplainer for other models
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                background
            )
    
    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a dataset.
        
        Args:
            X: Feature data
            
        Returns:
            SHAP values array (2D: samples x features)
        """
        if isinstance(self.explainer, shap.TreeExplainer):
            shap_values = self.explainer.shap_values(X)
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # List of arrays for each class - take positive class
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                else:
                    shap_values = shap_values[0]
            elif hasattr(shap_values, 'values'):
                # SHAP Explanation object
                shap_values = shap_values.values
            # Handle 3D array (samples, features, classes)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]  # Take positive class
        else:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            elif hasattr(shap_values, 'values'):
                shap_values = shap_values.values
        
        # Ensure 2D
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        
        self.shap_values = shap_values
        return shap_values
    
    def get_feature_importance(self, shap_values: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.
        
        Args:
            shap_values: Optional pre-computed SHAP values
            
        Returns:
            DataFrame with feature importance ranking
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        # Ensure 2D for mean calculation
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # Take positive class
        
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        # Flatten if needed
        if importance.ndim > 1:
            importance = importance.flatten()
        
        df = pd.DataFrame({
            'feature': self.feature_names[:len(importance)],
            'mean_abs_shap': importance
        })
        
        return df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    
    def plot_summary(
        self, 
        X: np.ndarray,
        shap_values: Optional[np.ndarray] = None,
        max_display: int = 15,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Create SHAP summary beeswarm plot.
        
        Args:
            X: Feature data
            shap_values: Optional pre-computed SHAP values
            max_display: Maximum features to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        fig, ax = plt.subplots(figsize=figsize)
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        return fig
    
    def plot_bar(
        self,
        shap_values: Optional[np.ndarray] = None,
        max_display: int = 15,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Create SHAP bar plot for feature importance.
        
        Args:
            shap_values: Optional pre-computed SHAP values
            max_display: Maximum features to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        importance_df = self.get_feature_importance(shap_values)[:max_display]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(
            importance_df['feature'], 
            importance_df['mean_abs_shap'],
            color='steelblue',
            edgecolor='black'
        )
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('Global Feature Importance (SHAP)')
        plt.tight_layout()
        return fig
    
    def plot_dependence(
        self,
        X: np.ndarray,
        feature: str,
        interaction_feature: Optional[str] = None,
        shap_values: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Create SHAP dependence plot.
        
        Args:
            X: Feature data
            feature: Feature to analyze
            interaction_feature: Feature for color-coding
            shap_values: Optional pre-computed SHAP values
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        feature_idx = self.feature_names.index(feature)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if interaction_feature:
            interaction_idx = self.feature_names.index(interaction_feature)
            shap.dependence_plot(
                feature_idx, shap_values, X,
                feature_names=self.feature_names,
                interaction_index=interaction_idx,
                show=False, ax=ax
            )
        else:
            shap.dependence_plot(
                feature_idx, shap_values, X,
                feature_names=self.feature_names,
                show=False, ax=ax
            )
        
        plt.tight_layout()
        return fig


class LocalExplainer:
    """
    Handles individual prediction explainability using SHAP and LIME.
    """
    
    def __init__(
        self, 
        model, 
        X_train: np.ndarray, 
        feature_names: List[str],
        class_names: List[str] = ['Good Credit', 'Bad Credit']
    ):
        """
        Initialize local explainer.
        
        Args:
            model: Trained classifier
            X_train: Training data
            feature_names: List of feature names
            class_names: Names for target classes
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.class_names = class_names
        
        # Create SHAP explainer
        model_name = type(self.model).__name__
        if model_name in ['RandomForestClassifier', 'GradientBoostingClassifier',
                           'DecisionTreeClassifier']:
            self.shap_explainer = shap.TreeExplainer(self.model)
        else:
            background = shap.sample(X_train, 100) if len(X_train) > 100 else X_train
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                background
            )
        
        # Create LIME explainer
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True
        )
    
    def explain_with_shap(
        self, 
        instance: np.ndarray,
        return_values: bool = True
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP.
        
        Args:
            instance: Single instance (1D or 2D array)
            return_values: Whether to return raw SHAP values
            
        Returns:
            Dictionary with explanation details
        """
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)
        
        # Get prediction
        prediction = self.model.predict(instance)[0]
        probability = self.model.predict_proba(instance)[0]
        
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(instance)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]
        elif hasattr(shap_values, 'values'):
            shap_values = shap_values.values
        
        # Handle 3D array (samples, features, classes)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # Take positive class
        
        # Ensure 1D for single instance
        shap_values_flat = shap_values.flatten()
        instance_flat = instance.flatten()
        
        # Match lengths (in case of encoding differences)
        min_len = min(len(self.feature_names), len(shap_values_flat), len(instance_flat))
        
        # Create feature contributions
        contributions = pd.DataFrame({
            'feature': self.feature_names[:min_len],
            'value': instance_flat[:min_len],
            'shap_value': shap_values_flat[:min_len]
        })
        contributions['abs_shap'] = contributions['shap_value'].abs()
        contributions = contributions.sort_values('abs_shap', ascending=False)
        
        result = {
            'prediction': self.class_names[prediction],
            'probability': probability[1],  # Probability of positive class (bad credit)
            'contributions': contributions,
            'top_positive': contributions[contributions['shap_value'] > 0].head(5),
            'top_negative': contributions[contributions['shap_value'] < 0].head(5)
        }
        
        if return_values:
            result['shap_values'] = shap_values
        
        return result
    
    def explain_with_lime(
        self, 
        instance: np.ndarray,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using LIME.
        
        Args:
            instance: Single instance (1D array)
            num_features: Number of features to include
            
        Returns:
            Dictionary with LIME explanation
        """
        if instance.ndim == 2:
            instance = instance.flatten()
        
        # Get LIME explanation
        exp = self.lime_explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=num_features
        )
        
        # Get prediction
        prediction = self.model.predict(instance.reshape(1, -1))[0]
        probability = self.model.predict_proba(instance.reshape(1, -1))[0]
        
        # Parse explanation
        explanation_list = exp.as_list()
        
        return {
            'prediction': self.class_names[prediction],
            'probability': probability[1],
            'explanation': explanation_list,
            'lime_object': exp
        }
    
    def plot_shap_waterfall(
        self, 
        instance: np.ndarray,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Create SHAP waterfall plot for single prediction.
        
        Args:
            instance: Single instance
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)
        
        shap_values = self.shap_explainer.shap_values(instance)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]
        elif hasattr(shap_values, 'values'):
            shap_values = shap_values.values
        
        # Handle 3D array
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        
        # Create SHAP Explanation object
        expected_value = self.shap_explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Manual waterfall plot
        values = shap_values.flatten()
        
        # Limit to number of features
        n_features = min(len(self.feature_names), len(values))
        values = values[:n_features]
        
        # Get top 10 indices
        sorted_idx = np.argsort(np.abs(values))[::-1][:min(10, n_features)]
        
        features = [self.feature_names[i] for i in sorted_idx]
        contributions = [values[i] for i in sorted_idx]
        
        colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in contributions]
        
        y_pos = range(len(features))
        ax.barh(y_pos, contributions, color=colors, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('SHAP Value (impact on model output)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('SHAP Waterfall Plot - Individual Prediction')
        
        plt.tight_layout()
        return fig
    
    def plot_lime_explanation(
        self, 
        lime_exp,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot LIME explanation as bar chart.
        
        Args:
            lime_exp: LIME explanation object
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        exp_list = lime_exp.as_list()
        
        features = [x[0] for x in exp_list]
        weights = [x[1] for x in exp_list]
        colors = ['#ff6b6b' if w > 0 else '#4ecdc4' for w in weights]
        
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = range(len(features))
        ax.barh(y_pos, weights, color=colors, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Weight')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('LIME Explanation - Individual Prediction')
        
        plt.tight_layout()
        return fig


def generate_explanation_text(
    shap_result: Dict,
    lime_result: Optional[Dict] = None
) -> str:
    """
    Generate plain-English explanation from SHAP/LIME results.
    
    Args:
        shap_result: Result from explain_with_shap
        lime_result: Optional result from explain_with_lime
        
    Returns:
        Human-readable explanation string
    """
    lines = []
    
    # Prediction summary
    pred = shap_result['prediction']
    prob = shap_result['probability']
    
    if prob > 0.7:
        risk_level = "HIGH"
    elif prob > 0.5:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
    
    lines.append(f"**Prediction**: {pred}")
    lines.append(f"**Risk Score**: {prob:.1%} ({risk_level} risk)")
    lines.append("")
    
    # Top risk factors
    lines.append("**Key Risk Factors (increasing risk):**")
    for _, row in shap_result['top_positive'].iterrows():
        feature = row['feature']
        value = row['value']
        impact = row['shap_value']
        lines.append(f"  • {feature} = {value:.2f} (impact: +{impact:.3f})")
    
    lines.append("")
    
    # Protective factors
    lines.append("**Protective Factors (decreasing risk):**")
    for _, row in shap_result['top_negative'].iterrows():
        feature = row['feature']
        value = row['value']
        impact = row['shap_value']
        lines.append(f"  • {feature} = {value:.2f} (impact: {impact:.3f})")
    
    return "\n".join(lines)
