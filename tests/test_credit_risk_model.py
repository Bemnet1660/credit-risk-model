"""
Unit tests for Credit Risk Prediction Model
"""

import pytest
import pandas as pd
import numpy as np
import os
from src.credit_risk_model import CreditRiskPredictor, ModelConfig
from src.data_generator import generate_credit_data


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return generate_credit_data(100, random_state=42)


@pytest.fixture
def model():
    """Create a model instance."""
    config = ModelConfig(n_estimators=10, max_depth=5)
    return CreditRiskPredictor(config)


def test_model_initialization(model):
    """Test that the model initializes correctly."""
    assert model.config.n_estimators == 10
    assert model.config.max_depth == 5
    assert model._is_trained is False
    assert model.model is None


def test_data_loading_validation(model, sample_data):
    """Test that data loading validates required columns."""
    # Save sample data
    sample_data.to_csv('test_data.csv', index=False)

    # Test that loading works
    loaded_data = model.load_data('test_data.csv')
    assert len(loaded_data) > 0

    # Test missing column raises error
    invalid_data = sample_data.drop('loan_amount', axis=1)
    invalid_data.to_csv('invalid_data.csv', index=False)
    with pytest.raises(ValueError):
        model.load_data('invalid_data.csv')

    # Cleanup
    os.remove('test_data.csv')
    os.remove('invalid_data.csv')


def test_split_data(model, sample_data):
    """Test data splitting."""
    model.load_data('test_data.csv')  # save first
    model.split_data(sample_data)
    assert model.X_train is not None
    assert model.X_test is not None
    assert len(model.X_train) > 0
    assert len(model.X_test) > 0
    # Cleanup
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')


def test_training_workflow(model, sample_data):
    """Test the complete training workflow."""
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']
    model.train(X, y)
    assert model._is_trained is True
    assert model.model is not None
    assert hasattr(model, 'metrics')


def test_prediction(model, sample_data):
    """Test that predictions work correctly."""
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']
    model.train(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_prediction_untrained(model, sample_data):
    """Test that prediction fails when model is untrained."""
    X = sample_data.drop('target', axis=1)
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_feature_importance(model, sample_data):
    """Test that feature importance works correctly."""
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']
    model.train(X, y)
    importance = model.get_feature_importance()
    assert len(importance) == len(X.columns)
    assert np.isclose(importance.sum(), 1.0)


def test_data_cleaning_removes_outliers(model):
    """Test that outlier removal works."""
    data = pd.DataFrame({
        'loan_amount': [1000, 2000, 3000, 1000000],  # outlier
        'credit_score': [650, 700, 750, 800],
        'income': [30000, 40000, 50000, 1000000],    # outlier
        'employment_length': [2, 5, 10, 15],
        'debt_to_income_ratio': [0.2, 0.3, 0.1, 0.5],
        'target': [0, 0, 1, 1]
    })
    cleaned = model._clean_data(data)
    # Outliers should be removed, so length < original
    assert len(cleaned) < len(data)


def test_predict_proba(model, sample_data):
    """Test probability predictions."""
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']
    model.train(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0) & (probs <= 1))
