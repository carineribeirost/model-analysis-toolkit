import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class ADCalculator:
    """
    Calculates the Applicability Domain (AD) using KNN Mahalanobis Distance and Leverage.
    """
    def __init__(self, training_data: pd.DataFrame, n_neighbors: int = 10):
        """
        Initialize the calculator with training data.
        
        Args:
            training_data: pd.DataFrame containing the descriptors for the training set.
            n_neighbors: Number of neighbors to use for KNN distance calculation.
        """
        self.training_data = training_data
        self.p = training_data.shape[1]  # Number of descriptors (features)
        self.n = training_data.shape[0]  # Number of samples
        self.n_neighbors = n_neighbors
        
        # Calculate Mahalanobis inverse covariance matrix (VI)
        cov = np.cov(training_data.T)
        # Use pseudoinverse for stability if the covariance is near-singular
        if np.linalg.cond(cov) < 1/np.finfo(cov.dtype).eps:
            self.vi = np.linalg.inv(cov)
        else:
            self.vi = np.linalg.pinv(cov)
        
        # Initialize KNN model with Mahalanobis metric
        # Use n_neighbors + 1 to allow skipping 'self' in training data
        self.knn_model = NearestNeighbors(
            metric='mahalanobis', 
            metric_params={'VI': self.vi}, 
            n_neighbors=self.n_neighbors + 1
        )
        self.knn_model.fit(training_data)
        
        # Precompute (X^T X)^-1 for leverage calculation (using Moore-Penrose pseudoinverse)
        self.xtx_inv = np.linalg.pinv(training_data.T @ training_data)
        
        # Precalculate thresholds based on training data
        self.knn_threshold = self._calculate_internal_knn_threshold()
        self.leverage_threshold = 3 * self.p / self.n

    def _calculate_internal_knn_threshold(self) -> float:
        """Calculates the KNN threshold: mean + 3*std of internal set distances."""
        distances, _ = self.knn_model.kneighbors(self.training_data)
        # Skip the first column as it's the distance to self (0.0)
        internal_knn = distances[:, 1:].mean(axis=1)
        return internal_knn.mean() + 3 * internal_knn.std()

    def calculate_metrics(self, data: pd.DataFrame, is_internal: bool = False):
        """
        Calculates KNN distance and Leverage for the given dataset.
        
        Args:
            data: pd.DataFrame of descriptors to evaluate.
            is_internal: Whether this is the training dataset (affects neighbor selection).
            
        Returns:
            tuple: (knn_distances, leverage_values)
        """
        # KNN Distances
        distances, _ = self.knn_model.kneighbors(data)
        if is_internal:
            # Exclude self distance
            knn_dist = distances[:, 1:].mean(axis=1)
        else:
            # Use all k neighbors
            knn_dist = distances[:, :self.n_neighbors].mean(axis=1)
            
        # Leverage Calculation
        X = data.values
        # h_i = diag(X (X^T X)^-1 X^T)
        leverage = np.sum(X @ self.xtx_inv * X, axis=1)
        
        return knn_dist, leverage

    def assess(self, data: pd.DataFrame, dataset_name: str = "Dataset", is_internal: bool = False) -> pd.DataFrame:
        """
        Assesses whether molecules are within the Applicability Domain.
        
        Returns a DataFrame with calculated metrics and flags.
        """
        knn_dist, leverage = self.calculate_metrics(data, is_internal=is_internal)
        
        knn_in = knn_dist <= self.knn_threshold
        lev_in = leverage <= self.leverage_threshold
        in_domain = knn_in & lev_in
        
        return pd.DataFrame({
            'Dataset': dataset_name,
            'KNN_Distance': knn_dist,
            'Leverage': leverage,
            'KNN_In_Domain': knn_in,
            'Leverage_In_Domain': lev_in,
            'In_Applicability_Domain': in_domain
        }, index=data.index)
