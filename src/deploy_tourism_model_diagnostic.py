
import os
import pandas as pd
import joblib
from huggingface_hub import HfApi, login, upload_file
import subprocess
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def manage_dependencies():
    try:
        import numpy
        import dill
        import sklearn
        logging.info(f"NumPy version: {numpy.__version__}, dill version: {dill.__version__}, scikit-learn version: {sklearn.__version__}")
    except ImportError:
        logging.warning("Installing required libraries...")
        subprocess.check_call(["pip", "install", "--no-cache-dir",
                              "numpy==1.26.4", "pandas==2.2.2", "scikit-learn==1.6.1",
                              "joblib==1.4.2", "dill==0.3.8", "huggingface_hub==0.23.0",
                              "flask==3.0.3", "waitress==3.0.0"])
        logging.info("Libraries installed successfully.")
    return True

def authenticate(hf_token):
    if not hf_token:
        logging.warning("No valid Hugging Face token provided.")
        return False
    try:
        login(token=hf_token, add_to_git_credential=False)
        logging.info("Authenticated with Hugging Face successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to authenticate with Hugging Face: {e}")
        return False

def create_dockerfile():
    dockerfile_content = """
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY model.joblib .
COPY columns.joblib .
COPY input_data.csv .
RUN ls -la /app
EXPOSE 7860
CMD ["waitress-serve", "--host=0.0.0.0", "--port=7860", "--threads=4", "--call", "app:app"]
"""
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    logging.info("Dockerfile created.")

def create_requirements():
    requirements_content = """
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.6.1
joblib==1.4.2
dill==0.3.8
flask==3.0.3
waitress==3.0.0
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    logging.info("requirements.txt created.")

def load_and_save_model(model_path):
    default_model_path = "/content/repo/models/model.joblib"
    default_columns_path = "/content/repo/models/columns.joblib"
    colab_model_path = "/content/models/best_rf_model.joblib"
    colab_columns_path = "/content/models/columns.joblib"
    model_path = model_path or default_model_path
    
    if os.path.exists(colab_model_path) and os.path.abspath(colab_model_path) != os.path.abspath(default_model_path):
        shutil.copy(colab_model_path, default_model_path)
        logging.info(f"Model copied from {colab_model_path} to {default_model_path}")
    elif os.path.exists(model_path):
        logging.info(f"Model already exists at {model_path}")
    else:
        logging.error(f"Model not found at {colab_model_path} or {model_path}")
        return False
    
    if os.path.exists(colab_columns_path) and os.path.abspath(colab_columns_path) != os.path.abspath(default_columns_path):
        shutil.copy(colab_columns_path, default_columns_path)
        logging.info(f"Columns copied from {colab_columns_path} to {default_columns_path}")
    elif os.path.exists(default_columns_path):
        logging.info(f"Columns already exist at {default_columns_path}")
    else:
        logging.error(f"Columns file not found at {colab_columns_path} or {default_columns_path}")
        return False
    
    if not os.path.exists(default_model_path) or not os.path.exists(default_columns_path):
        logging.error("Model or columns files not found in deployment directory")
        return False
    return True

def prepare_sample_data():
    from datasets import load_dataset
    dataset = load_dataset("Shramik121/tourism-split-dataset")
    sample_df = pd.DataFrame(dataset['test']).sample(3)
    sample_df.drop(columns=['ProdTaken'], inplace=True, errors='ignore')
    required_columns = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
                       'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore', 
                       'NumberOfChildrenVisiting', 'MonthlyIncome', 'TypeofContact', 
                       'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 
                       'Designation', 'CityTier']
    sample_df = sample_df[required_columns]
    sample_df.to_csv("input_data.csv", index=False)
    logging.info("Input data saved to input_data.csv with required columns")
    return sample_df

def create_hosting_script():
    hosting_script_content = """
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
"""
    with open("app.py", "w") as f:
        f.write(hosting_script_content)
    logging.info("app.py created.")

def upload_to_huggingface(space_name):
    try:
        api = HfApi()
        api.create_repo(repo_id=space_name, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
        logging.info(f"Created or verified Space: {space_name}")
        
        required_files = ['app.py', 'model.joblib', 'columns.joblib', 'input_data.csv', 'requirements.txt', 'Dockerfile']
        for file in required_files:
            if not os.path.exists(file):
                logging.error(f"Required file {file} not found")
                return False
            logging.info(f"File {file} exists")
        
        for file in required_files:
            upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=space_name,
                repo_type="space",
                commit_message=f"Upload {file} to Hugging Face Space"
            )
            logging.info(f"Uploaded {file} to {space_name}")
        
        logging.info(f"Successfully uploaded files to https://huggingface.co/spaces/{space_name}")
        return True
    except Exception as e:
        logging.error(f"Failed to upload to Hugging Face: {e}")
        return False

if __name__ == "__main__":
    if manage_dependencies():
        hf_token = os.getenv("HF_TOKEN")
        authenticated = authenticate(hf_token) if hf_token else False
        create_dockerfile()
        create_requirements()
        if load_and_save_model(os.getenv("MODEL_PATH")):
            prepare_sample_data()
            create_hosting_script()
            if authenticated:
                space_name = os.getenv("SPACE_NAME", "Shramik121/tourism-rf-model")
                upload_to_huggingface(space_name)
            else:
                logging.warning("Skipping upload due to authentication failure")
        else:
            logging.warning("Skipping data preparation and upload due to model loading failure")
