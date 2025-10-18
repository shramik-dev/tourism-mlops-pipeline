
import pytest
import joblib
import os

def test_model_exists():
    assert os.path.exists('models/model.joblib'), "Model file not found"

def test_columns_exists():
    assert os.path.exists('models/columns.joblib'), "Columns file not found"

def test_model_loads():
    model = joblib.load('models/model.joblib')
    assert model is not None, "Failed to load model"
