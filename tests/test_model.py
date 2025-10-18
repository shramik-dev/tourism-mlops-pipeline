
import pytest
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_model_exists():
    model_path = os.getenv('MODEL_PATH', 'models/model.joblib')
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    logging.info(f"Model exists at {model_path}")

def test_columns_exists():
    columns_path = 'models/columns.joblib'
    assert os.path.exists(columns_path), f"Columns file not found at {columns_path}"
    logging.info(f"Columns file exists at {columns_path}")

def test_model_loads():
    model_path = os.getenv('MODEL_PATH', 'models/model.joblib')
    model = joblib.load(model_path)
    assert model is not None, "Failed to load model"
    logging.info("Model loaded successfully")
