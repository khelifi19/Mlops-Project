import os
import pytest
import sys

# 🔹 Fix ImportError: Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Import functions to test
from model_pipeline import prepare_data, train_model, evaluate_model, load_model


def test_prepare_data():
    """Test if data preparation loads the dataset correctly."""
    X, y, _, _ = prepare_data()
    assert X.shape[0] > 0  # Ensure features are loaded
    assert y.shape[0] > 0  # Ensure target is loaded


def test_train_model():
    """Test if training returns a trained model."""
    X, y, _, _ = prepare_data()
    model = train_model(X, y)
    assert model is not None


def test_load_model():
    """Test if model loads correctly after training."""
    model, scaler, pca = load_model()
    assert model is not None  # Ensure model exists
    assert scaler is not None  # Ensure scaler exists
    assert pca is not None  # Ensure PCA exists


def test_evaluate_model():
    """Test if model evaluation returns a valid accuracy score."""
    X, y, _, _ = prepare_data()
    model, _, _ = load_model()
    accuracy = evaluate_model(model, X, y)
    assert 0.0 <= accuracy <= 1.0  # Accuracy should be between 0 and 1
