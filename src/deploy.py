
import os
import pandas as pd
import joblib
from huggingface_hub import HfApi, login, upload_file
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dockerfile():
    with open("Dockerfile", "w") as f:
        f.write('''
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY model.joblib .
COPY columns.joblib .
COPY input_data.csv .
EXPOSE 7860
CMD ["waitress-serve", "--host=0.0.0.0", "--port=7860", "--threads=4", "--call", "app:app"]
''')
    logging.info("Dockerfile created")

def create_requirements():
    with open("requirements.txt", "w") as f:
        f.write('''
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.6.1
joblib==1.4.2
dill==0.3.8
flask==3.0.3
waitress==3.0.0
''')
    logging.info("requirements.txt created")

def create_app():
    with open("app.py", "w") as f:
        f.write('''
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
base_dir = '/app' if os.path.exists('/app') else os.getcwd()
model = joblib.load(os.path.join(base_dir, "model.joblib"))
columns = joblib.load(os.path.join(base_dir, "columns.joblib"))

required_columns = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
                   'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore', 
                   'NumberOfChildrenVisiting', 'MonthlyIncome', 'TypeofContact', 
                   'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 
                   'Designation', 'CityTier']

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = [data] if isinstance(data, dict) else data
        input_df = pd.DataFrame(input_data)
        num_cols = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
                    'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore', 
                    'NumberOfChildrenVisiting', 'MonthlyIncome']
        cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
                    'MaritalStatus', 'Designation', 'CityTier']
        for col in required_columns:
            if col not in input_df.columns:
                input_df[col] = 0.0 if col in num_cols else 'Unknown'
        input_df[num_cols] = input_df[num_cols].astype(float).fillna(input_df[num_cols].median())
        input_df[cat_cols] = input_df[cat_cols].fillna('Unknown')
        input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
        for col in columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)
        prediction = model.predict(input_encoded)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=7860, threads=4)
''')
    logging.info("app.py created")

def prepare_sample_data():
    dataset = load_dataset("Shramik121/tourism-split-dataset")
    sample_df = pd.DataFrame(dataset['test']).sample(2)  # Reduced sample size
    sample_df.drop(columns=['ProdTaken'], inplace=True, errors='ignore')
    required_columns = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
                       'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore', 
                       'NumberOfChildrenVisiting', 'MonthlyIncome', 'TypeofContact', 
                       'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 
                       'Designation', 'CityTier']
    num_cols = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
                'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore', 
                'NumberOfChildrenVisiting', 'MonthlyIncome']
    cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
                'MaritalStatus', 'Designation', 'CityTier']
    for col in required_columns:
        if col not in sample_df.columns:
            sample_df[col] = 0.0 if col in num_cols else 'Unknown'
    sample_df = sample_df[required_columns]
    sample_df.to_csv("input_data.csv", index=False)
    logging.info("input_data.csv created")

def deploy():
    login(token=os.getenv("HF_TOKEN"))
    space_name = os.getenv("SPACE_NAME", "Shramik121/tourism-rf-model")
    api = HfApi()
    api.create_repo(repo_id=space_name, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    files = ['app.py', 'model.joblib', 'columns.joblib', 'input_data.csv', 'requirements.txt', 'Dockerfile']
    for file in files:
        if os.path.exists(file):
            upload_file(path_or_fileobj=file, path_in_repo=file, repo_id=space_name, repo_type="space")
            logging.info(f"Uploaded {file} to {space_name}")
        else:
            logging.error(f"File {file} not found")
            raise FileNotFoundError(f"File {file} not found")

if __name__ == "__main__":
    create_dockerfile()
    create_requirements()
    create_app()
    prepare_sample_data()
    deploy()
