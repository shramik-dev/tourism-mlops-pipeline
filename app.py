
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
    model = joblib.load(os.path.join(base_dir, "model.joblib"))
    columns = joblib.load(os.path.join(base_dir, "columns.joblib"))
    logger.info("Model and columns loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model or columns: {e}")
    raise

required_columns = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
                   'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore', 
                   'NumberOfChildrenVisiting', 'MonthlyIncome', 'TypeofContact', 
                   'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 
                   'Designation', 'CityTier']

@app.route('/', methods=['GET'])
def index():
    logger.info("Root endpoint called")
    return jsonify({'status': 'ok'})

@app.route('/health', methods=['GET'])
def health():
    logger.info("Health check endpoint called")
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        logger.info(f"Predict endpoint called with data: {data}")
        input_df = pd.DataFrame(data)
        
        # Validate input columns
        missing_cols = [col for col in required_columns if col not in input_df.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
        
        # Handle missing values
        num_cols = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
                    'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore', 
                    'NumberOfChildrenVisiting', 'MonthlyIncome']
        cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
                    'MaritalStatus', 'Designation', 'CityTier']
        input_df[num_cols] = input_df[num_cols].fillna(input_df[num_cols].median())
        input_df[cat_cols] = input_df[cat_cols].fillna('Unknown')
        
        # Encode input data
        input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(input_encoded)
        logger.info(f"Prediction made: {prediction.tolist()}")
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    from waitress import serve
    logger.info("Starting Waitress server on port 7860")
    serve(app, host='0.0.0.0', port=7860, threads=4)
