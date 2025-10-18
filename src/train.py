
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from datasets import load_dataset
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    dataset = load_dataset("Shramik121/tourism-split-dataset")
    data = pd.DataFrame(dataset['train'])
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    
    required_columns = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
                       'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore', 
                       'NumberOfChildrenVisiting', 'MonthlyIncome', 'TypeofContact', 
                       'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 
                       'Designation', 'CityTier']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        logging.error(f"Training data missing columns: {missing_cols}")
        raise ValueError(f"Training data missing columns: {missing_cols}")
    
    num_cols = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
                'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore', 
                'NumberOfChildrenVisiting', 'MonthlyIncome']
    cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
                'MaritalStatus', 'Designation', 'CityTier']
    
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    data[cat_cols] = data[cat_cols].fillna('Unknown')
    
    X = data.drop(columns=['ProdTaken'])
    y = data['ProdTaken']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])
    
    pipeline.fit(X, y)
    
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    columns = X_encoded.columns.tolist()
    os.makedirs('models', exist_ok=True)
    joblib.dump(columns, 'models/columns.joblib')
    model_path = os.getenv('MODEL_OUTPUT', 'models/model.joblib')
    joblib.dump(pipeline, model_path)
    
    # Log model to MLflow with input example
    mlflow.set_tracking_uri("file://./mlruns")
    mlflow.set_experiment("Tourism_Package_Prediction")
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.log_params({"random_state": 42})
        input_example = X.iloc[:1]
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="random_forest_model",
            input_example=input_example
        )
        mlflow.log_artifact(model_path)
    logging.info("Model and columns saved to models/")

if __name__ == "__main__":
    train_model()
