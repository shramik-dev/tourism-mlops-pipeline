
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

base_dir = '/app' if os.path.exists('/app') else os.getcwd()
logger.info(f"Using base directory: {base_dir}, contents: {os.listdir(base_dir)}")

try:
    model_path = os.path.join(base_dir, "model.joblib")
    columns_path = os.path.join(base_dir, "columns.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.joblib not found at {model_path}")
    if not os.path.exists(columns_path):
        raise FileNotFoundError(f"columns.joblib not found at {columns_path}")
    model = joblib.load(model_path)
    columns = joblib.load(columns_path)
    logger.info(f"Model and columns loaded successfully from {base_dir}")
except Exception as e:
    logger.error(f"Failed to load model or columns: {e}")
    raise

@app.route('/', methods=['GET'])
def index():
    logger.info("Root endpoint called")
    return jsonify({'status': 'ok'})

@app.route('/health', methods=['GET'])
def health():
    logger.info("Health check endpoint called")
    if 'model' in globals() and 'columns' in globals():
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        logger.info(f"Predict endpoint called with data: {data}")
        if not isinstance(data, dict) or not data:
            return jsonify({'error': 'Invalid input data format. Expected a dictionary with list values.'}), 400
        input_df = pd.DataFrame(data)
        categorical_columns = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation', 'CityTier']
        input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)
        prediction = model.predict(input_encoded)
        logger.info(f"Prediction made: {prediction.tolist()}")
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': str(e)}), 400
