import pytest
import pandas as pd
import numpy as np
from src.applicability_domain.calculator import ADCalculator

@pytest.fixture
def sample_data():
    """Creates a simple synthetic dataset for testing."""
    np.random.seed(42)
    # 100 samples, 3 descriptors
    data = np.random.normal(0, 1, (100, 3))
    return pd.DataFrame(data, columns=['desc1', 'desc2', 'desc3'])

def test_calculator_initialization(sample_data):
    """Test if calculator initializes correctly and computes thresholds."""
    calc = ADCalculator(sample_data, n_neighbors=5)
    
    assert calc.p == 3
    assert calc.n == 100
    assert calc.vi.shape == (3, 3)
    assert calc.knn_threshold > 0
    # Leverage threshold for p=3, n=100 should be 3*3/100 = 0.09
    assert calc.leverage_threshold == pytest.approx(0.09)

def test_knn_distance_internal(sample_data):
    """Test KNN distances for internal data (should exclude self)."""
    calc = ADCalculator(sample_data, n_neighbors=5)
    knn_dist, _ = calc.calculate_metrics(sample_data, is_internal=True)
    
    assert len(knn_dist) == 100
    assert np.all(knn_dist > 0)  # Distances should be > 0 if self is excluded

def test_leverage_calculation(sample_data):
    """Test leverage calculation manually for a known point."""
    calc = ADCalculator(sample_data)
    _, leverage = calc.calculate_metrics(sample_data)
    
    # The sum of leverage values should be equal to p
    assert np.sum(leverage) == pytest.approx(calc.p)

def test_assessment_logic(sample_data):
    """Test the AD assessment flags."""
    calc = ADCalculator(sample_data)
    
    # Create an outlier point (far from 0,0,0)
    outlier = pd.DataFrame([[10, 10, 10]], columns=['desc1', 'desc2', 'desc3'])
    
    results = calc.assess(outlier, dataset_name="Outlier", is_internal=False)
    
    assert results.iloc[0]['In_Applicability_Domain'] == False
    assert results.iloc[0]['KNN_In_Domain'] == False or results.iloc[0]['Leverage_In_Domain'] == False

def test_internal_assessment_consistency(sample_data):
    """Test that the majority of training data is usually within domain."""
    calc = ADCalculator(sample_data)
    results = calc.assess(sample_data, is_internal=True)
    
    # For a normal distribution, most points should be within mean + 3*std
    assert results['KNN_In_Domain'].sum() > 90 
