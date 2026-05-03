import pickle
import numpy as np
import pandas as pd
import shap
from typing import List, Union, Optional

class SHAPExplainer:
    """
    Handles SHAP value calculations for single models or ensembles.
    """
    def __init__(self, model_paths: List[str], task: str = "classification"):
        """
        Initialize with paths to pickled models.
        
        Args:
            model_paths: List of paths to .pkl files.
            task: "classification" or "regression".
        """
        self.model_paths = model_paths
        self.task = task.lower()
        self.models = self._load_models()
        self.feature_names = None

    def _load_models(self):
        models = []
        for path in self.model_paths:
            with open(path, "rb") as f:
                models.append(pickle.load(f))
        return models

    def get_ensemble_explanation(self, X: pd.DataFrame) -> shap.Explanation:
        """
        Calculates and averages SHAP values across all models in the ensemble.
        
        Args:
            X: Input dataset (features only).
            
        Returns:
            shap.Explanation: Averaged explanation object.
        """
        self.feature_names = X.columns.tolist()
        all_shap_values = []
        base_values = []

        for model in self.models:
            explainer = shap.Explainer(model, X)
            explanation = explainer(X)
            
            # Extract values
            values = explanation.values
            bv = explanation.base_values

            # Handle Classification (Multiclass/Binary)
            # shap returns (samples, features, classes) for classifiers
            if self.task == "classification" and len(values.shape) == 3:
                # We target the positive class (index 1)
                values = values[:, :, 1]
                if isinstance(bv, np.ndarray) and len(bv.shape) > 1:
                    bv = bv[:, 1]

            all_shap_values.append(values)
            base_values.append(bv)

        # Average results
        mean_shap_values = np.mean(all_shap_values, axis=0)
        mean_base_values = np.mean(base_values, axis=0)

        # Create a combined explanation object using the first model's structure as a template
        combined_explanation = shap.Explanation(
            values=mean_shap_values,
            base_values=mean_base_values,
            data=X.values,
            feature_names=self.feature_names
        )

        return combined_explanation

    @staticmethod
    def normalize_explanation(explanation: shap.Explanation) -> shap.Explanation:
        """
        Standardizes SHAP values (mean 0, std 1) per feature.
        Useful for visualizing impact on unbalanced data.
        """
        values = explanation.values
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        std = np.maximum(std, 1e-8)  # Prevent division by zero

        normalized_values = (values - mean) / std
        
        return shap.Explanation(
            values=normalized_values,
            base_values=explanation.base_values,
            data=explanation.data,
            feature_names=explanation.feature_names
        )
