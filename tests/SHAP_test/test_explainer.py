import pytest
import pandas as pd
import numpy as np
import pickle
import os
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.shap_analysis.explainer import SHAPExplainer

@pytest.fixture
def temp_models(tmp_path):
    """Creates small dummy models for testing ensemble logic."""
    np.random.seed(42)
    X = np.random.rand(20, 5)
    y_clf = (X[:, 0] > 0.5).astype(int)
    y_reg = X[:, 0] * 10
    
    # Classification Model
    clf = RandomForestClassifier(n_estimators=2, max_depth=2)
    clf.fit(X, y_clf)
    clf_path = os.path.join(tmp_path, "clf.pkl")
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)
        
    # Regression Model
    reg = RandomForestRegressor(n_estimators=2, max_depth=2)
    reg.fit(X, y_reg)
    reg_path = os.path.join(tmp_path, "reg.pkl")
    with open(reg_path, "wb") as f:
        pickle.dump(reg, f)
        
    return {"clf": clf_path, "reg": reg_path, "data": pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])}

def test_ensemble_averaging_classification(temp_models):
    """Test if averaging works and class 1 is correctly selected."""
    # Use the same model twice as a simple ensemble test
    explainer = SHAPExplainer([temp_models["clf"], temp_models["clf"]], task="classification")
    exp = explainer.get_ensemble_explanation(temp_models["data"])
    
    assert isinstance(exp, shap.Explanation)
    # Original SHAP for RF classifier is (samples, features, classes)
    # Our explainer should have extracted class 1, resulting in (samples, features)
    assert exp.values.shape == (20, 5)
    assert exp.feature_names == temp_models["data"].columns.tolist()

def test_ensemble_averaging_regression(temp_models):
    """Test if averaging works for regression."""
    explainer = SHAPExplainer([temp_models["reg"], temp_models["reg"]], task="regression")
    exp = explainer.get_ensemble_explanation(temp_models["data"])
    
    assert exp.values.shape == (20, 5)

def test_normalization_logic():
    """Test the static normalization method."""
    # Create a mock explanation
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    mock_exp = shap.Explanation(
        values=values,
        base_values=np.zeros(2),
        data=np.zeros((2, 2)),
        feature_names=["a", "b"]
    )
    
    norm_exp = SHAPExplainer.normalize_explanation(mock_exp)
    
    # Mean of each column in norm_exp should be approx 0
    assert np.mean(norm_exp.values, axis=0) == pytest.approx([0, 0], abs=1e-7)
    # Std of each column should be 1
    assert np.std(norm_exp.values, axis=0) == pytest.approx([1, 1], abs=1e-7)

def test_feature_names_persistence(temp_models):
    """Ensure feature names are carried over."""
    explainer = SHAPExplainer([temp_models["clf"]], task="classification")
    exp = explainer.get_ensemble_explanation(temp_models["data"])
    assert exp.feature_names == ["f0", "f1", "f2", "f3", "f4"]
