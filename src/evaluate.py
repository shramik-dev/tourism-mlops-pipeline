
import os
import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline # Import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model():
    model_path = os.getenv('MODEL_PATH', 'models/model.joblib')
    test_data_path = os.getenv('TEST_DATA', 'data/test.csv')
    evaluation_output_path = 'evaluation_results.json' # Define output path

    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        results = {'error': f'Model file not found at {model_path}'}
        with open(evaluation_output_path, 'w') as f:
            json.dump(results, f)
        return

    if not os.path.exists(test_data_path):
        logging.error(f"Test data file not found at {test_data_path}")
        results = {'error': f'Test data file not found at {test_data_path}'}
        with open(evaluation_output_path, 'w') as f:
            json.dump(results, f)
        return

    model = joblib.load(model_path)
    data = pd.read_csv(test_data_path)

    required_columns = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                       'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore',
                       'NumberOfChildrenVisiting', 'MonthlyIncome', 'TypeofContact',
                       'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus',
                       'Designation', 'CityTier', 'ProdTaken'] # Include target for validation
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        logging.error(f"Test data missing columns: {missing_cols}")
        raise ValueError(f"Test data missing columns: {missing_cols}")

    # Apply the same preprocessing steps as in train.py to the test data
    num_cols = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore',
                'NumberOfChildrenVisiting', 'MonthlyIncome']
    cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
                'MaritalStatus', 'Designation', 'CityTier']

    # Handle missing values (consistent with training)
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    data[cat_cols] = data[cat_cols].fillna('Unknown')

    X_test = data.drop(columns=['ProdTaken'])
    y_test = data['ProdTaken']

    # Ensure the loaded model is a pipeline and can process the raw X_test
    if isinstance(model, Pipeline):
         predictions = model.predict(X_test)
    else:
         # If the loaded model is just the classifier, apply preprocessing manually
         logging.warning("Loaded model is not a pipeline. Applying preprocessing manually.")
         preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ],
            remainder='passthrough'
         )
         preprocessor.fit(X_test) # Fit preprocessor on test data before transforming
         X_test_processed = preprocessor.transform(X_test)
         predictions = model.predict(X_test_processed)


    accuracy = accuracy_score(y_test, predictions)
    # Assuming binary classification, use F1 score for evaluation
    try:
        f1 = f1_score(y_test, predictions)
    except ValueError:
        logging.warning("Could not calculate F1 score, target might be single class.")
        f1 = None


    results = {
        'accuracy': accuracy,
        'f1_score': f1
        }
    with open(evaluation_output_path, 'w') as f:
        json.dump(results, f)

    logging.info(f"Model Accuracy: {accuracy}")
    if f1 is not None:
        logging.info(f"Model F1 Score: {f1}")


if __name__ == "__main__":
    evaluate_model()
