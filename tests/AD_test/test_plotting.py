import pytest
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for testing
from src.applicability_domain.plotting import ADPlotter

@pytest.fixture
def mock_results():
    data = {
        'Dataset': ['Internal']*50 + ['External']*50,
        'KNN_Distance': np.random.rand(100),
        'Leverage': np.random.rand(100),
        'In_Applicability_Domain': [True]*80 + [False]*20
    }
    return pd.DataFrame(data)

def test_ad_plots_generation(mock_results, tmp_path):
    plotter = ADPlotter()
    
    dist_path = os.path.join(tmp_path, "dist.png")
    bar_path = os.path.join(tmp_path, "bar.png")
    williams_path = os.path.join(tmp_path, "williams.png")
    
    plotter.plot_distributions(mock_results, 0.5, 0.5, save_path=dist_path)
    plotter.plot_summary_bar(mock_results, save_path=bar_path)
    plotter.plot_williams_ad(mock_results, 0.5, 0.5, save_path=williams_path)
    
    assert os.path.exists(dist_path)
    assert os.path.exists(bar_path)
    assert os.path.exists(williams_path)
