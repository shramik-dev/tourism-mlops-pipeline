
import os
import pandas as pd
import joblib
from huggingface_hub import HfApi, login, upload_folder
import subprocess
import shutil
import logging
from google.colab import userdata # Import userdata for local testing if needed

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
                              "joblib==1.4.2", "dill==0.3.8", "huggingface-hub==0.23.0",
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
CMD ["waitress-serve", "--host=0.0.0.0", "--port=7860", "--threads=4", "app:app"]
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
huggingface_hub==0.23.0
flask==3.0.3
waitress==3.0.0
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    logging.info("requirements.txt created.")

def load_and_save_model(model_path):
    default_model_path = "models/model.joblib"
    default_columns_path = "models/columns.joblib"
    colab_model_path = "/content/models/best_rf_model.joblib"
    colab_columns_path = "/content/models/columns.joblib"
    model_path = model_path or default_model_path

    # Determine the source paths based on where the script is run
    if os.path.exists(colab_model_path) and os.path.exists(colab_columns_path):
        src_model = colab_model_path
        src_columns = colab_columns_path
        logging.info("Using model and columns from Colab specific path.")
    elif os.path.exists(default_model_path) and os.path.exists(default_columns_path):
        src_model = default_model_path
        src_columns = default_columns_path
        logging.info("Using model and columns from default models directory.")
    else:
        logging.error(f"Model or columns files not found at {colab_model_path}, {colab_columns_path}, {default_model_path}, or {default_columns_path}")
        return False

    # Define the destination paths in the current directory
    dest_model = "model.joblib"
    dest_columns = "columns.joblib"

    try:
        # Copy model file, avoiding SameFileError
        if not os.path.exists(dest_model) or not os.path.samefile(src_model, dest_model):
             shutil.copy(src_model, dest_model)
             logging.info(f"Model copied from {src_model} to {dest_model}")
        else:
            logging.info(f"Model source and destination are the same ({src_model}), skipping copy.")

        # Copy columns file, avoiding SameFileError
        if not os.path.exists(dest_columns) or not os.path.samefile(src_columns, dest_columns):
            shutil.copy(src_columns, dest_columns)
            logging.info(f"Columns copied from {src_columns} to {dest_columns}")
        else:
             logging.info(f"Columns source and destination are the same ({src_columns}), skipping copy.")

    except Exception as e:
        logging.error(f"Error during file copy: {e}")
        return False


    # Verify both files exist in the deployment directory
    if not os.path.exists(dest_model) or not os.path.exists(dest_columns):
        logging.error("Model or columns files not found in deployment directory after copy attempt.")
        return False
    return True

def prepare_sample_data():
    from datasets import load_dataset
    try:
        dataset = load_dataset("Shramik121/tourism-split-dataset")
        sample_df = pd.DataFrame(dataset['test']).sample(min(3, len(dataset['test'])))
        sample_df.drop(columns=['ProdTaken', 'Unnamed: 0', '__index_level_0__'], inplace=True, errors='ignore') # Drop unnecessary columns
        sample_df.to_csv("input_data.csv", index=False)
        logging.info("Input data saved to input_data.csv")
    except Exception as e:
        logging.error(f"Failed to prepare sample data: {e}")
        # Create a dummy sample data if loading fails
        sample_inputs = {
            'Age': [41.0], 'TypeofContact': ['Self Enquiry'], 'CityTier': [3],
            'DurationOfPitch': [6.0], 'Occupation': ['Salaried'], 'Gender': ['Female'],
            'NumberOfPersonVisiting': [3], 'NumberOfFollowups': [3.0],
            'ProductPitched': ['Deluxe'], 'PreferredPropertyStar': [3.0],
            'MaritalStatus': ['Single'], 'NumberOfTrips': [1.0], 'Passport': [1],
            'PitchSatisfactionScore': [2], 'OwnCar': [1],
            'NumberOfChildrenVisiting': [0.0], 'Designation': ['Manager'],
            'MonthlyIncome': [20993.0]
        }
        input_df = pd.DataFrame(sample_inputs)
        input_df.to_csv("input_data.csv", index=False)
        logging.warning("Using dummy sample data due to loading failure.")


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

# Determine the base directory based on the environment
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
    # In a real application, you might return an error response or have a fallback
    raise # Re-raise the exception to indicate a critical startup failure

@app.route('/', methods=['GET'])
def index():
    logger.info("Root endpoint called")
    return jsonify({'status': 'ok'})

@app.route('/health', methods=['GET'])
def health():
    logger.info("Health check endpoint called")
    # Check if the model and columns are loaded
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

        # Ensure all expected columns are present and in the correct order
        # This requires knowledge of the columns used during training preprocessing
        categorical_columns = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation', 'CityTier']
        # Apply the same one-hot encoding and reindexing as in training
        input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)

        # Reindex to match the training columns, filling missing with 0
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

        prediction = model.predict(input_encoded)
        logger.info(f"Prediction made: {prediction.tolist()}")
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': str(e)}), 400

# Note: waitress runs this app in Docker; don't call app.run()
"""
    with open("app.py", "w") as f:
        f.write(hosting_script_content)
    logging.info("app.py created.")

def upload_to_huggingface(space_name):
    try:
        api = HfApi(token=os.getenv("HF_TOKEN"))
        # Create the Space if it doesn't exist
        api.create_repo(
            repo_id=space_name,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True
        )
        logging.info(f"Created or verified Space: {space_name}")

        # Verify files to be uploaded
        required_files = ['app.py', 'model.joblib', 'columns.joblib', 'input_data.csv', 'requirements.txt', 'Dockerfile']
        logging.info("Checking required files for upload: %s", required_files)
        for file in required_files:
            if not os.path.exists(file):
                logging.error(f"Required file {file} not found in deployment directory")
                return False
            else:
                logging.info(f"File {file} exists in deployment directory")

        # Upload files to the Space
        upload_folder(
            folder_path=".",
            repo_id=space_name,
            repo_type="space",
            path_in_repo="",
            commit_message="Upload model and application files to Hugging Face Space"
        )
        logging.info(f"Successfully uploaded files to https://huggingface.co/spaces/{space_name}")
        return True
    except Exception as e:
        logging.error(f"Failed to upload to Hugging Face: {e}")
        return False

if __name__ == "__main__":
    # Get HF_TOKEN from environment or Colab secrets for local testing
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        try:
            hf_token = userdata.get('HF_TOKEN')
            logging.info("Retrieved HF_TOKEN from Colab secrets.")
        except:
             logging.warning("HF_TOKEN not found in environment variables or Colab secrets.")
             hf_token = None


    if manage_dependencies():
        authenticated = authenticate(hf_token) if hf_token else False
        create_dockerfile()
        create_requirements()
        # No need to pass model_path from env here, load_and_save_model handles finding it
        if load_and_save_model(None): # Pass None to let the function find the model
            prepare_sample_data()
            create_hosting_script()
            if authenticated:
                space_name = os.getenv("SPACE_NAME", "Shramik121/tourism-rf-model")
                upload_to_huggingface(space_name)
            else:
                logging.warning("Skipping upload to Hugging Face due to authentication failure.")
        else:
            logging.warning("Skipping data preparation, hosting script, and upload due to model loading failure.")
    else:
        logging.warning("Skipping execution due to dependency issues.")

