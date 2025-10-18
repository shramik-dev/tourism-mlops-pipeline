
import pandas as pd
import joblib
import os
import logging
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    # Set up MLflow tracking directory
    mlruns_dir = os.path.join(os.getcwd(), 'mlruns')
    os.makedirs(mlruns_dir, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlruns_dir}"
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    logging.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

    # Set MLflow experiment
    mlflow.set_experiment("Tourism_Package_Prediction")

    # Load dataset
    dataset = load_dataset("Shramik121/tourism-split-dataset")
    data = pd.DataFrame(dataset['train'])
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)

    # Define features
    required_columns = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                       'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore',
                       'NumberOfChildrenVisiting', 'MonthlyIncome', 'TypeofContact',
                       'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus',
                       'Designation', 'CityTier', 'ProdTaken']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        logging.error(f"Training data missing columns: {missing_cols}")
        raise ValueError(f"Training data missing columns: {missing_cols}")

    num_cols = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore',
                'NumberOfChildrenVisiting', 'MonthlyIncome']
    cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
                'MaritalStatus', 'Designation', 'CityTier']

    # Handle missing values
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    data[cat_cols] = data[cat_cols].fillna('Unknown')

    X = data.drop(columns=['ProdTaken'])
    y = data['ProdTaken']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])

    # Model pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train model
    pipeline.fit(X, y)

    # Save column names after preprocessing
    dummy_df = pd.DataFrame(columns=X.columns)
    preprocessor.fit(dummy_df)
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)
    columns = feature_names

    # Save model and columns
    os.makedirs('models', exist_ok=True)
    model_path = os.getenv('MODEL_OUTPUT', 'models/model.joblib')
    joblib.dump(pipeline, model_path)
    joblib.dump(columns, 'models/columns.joblib')

    # Log to MLflow
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.log_params({"random_state": 42})
        input_example = X.iloc[:1]
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="random_forest_model",
            input_example=input_example
        )
        mlflow.log_artifact(model_path)
        mlflow.log_artifact('models/columns.joblib')
    logging.info(f"Model saved to {model_path}, columns saved to models/columns.joblib")

if __name__ == "__main__":
    train_model()
