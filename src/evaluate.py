
import os
import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model():
    model_path = os.getenv('MODEL_PATH', 'models/model.joblib')
    test_data_path = os.getenv('TEST_DATA', 'data/test.csv')
    evaluation_output_path = 'evaluation_results.json'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        results = {'error': f'Model file not found at {model_path}'}
        with open(evaluation_output_path, 'w') as f:
            json.dump(results, f)
        return
    if not os.path.exists(test_data_path):
        print(f"Error: Test data file not found at {test_data_path}")
        results = {'error': f'Test data file not found at {test_data_path}'}
        with open(evaluation_output_path, 'w') as f:
            json.dump(results, f)
        return
    model = joblib.load(model_path)
    data = pd.read_csv(test_data_path)
    num_cols = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore',
                'NumberOfChildrenVisiting', 'MonthlyIncome']
    cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
                'MaritalStatus', 'Designation', 'CityTier']
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    data[cat_cols] = data[cat_cols].fillna('Unknown')
    X_test = data.drop(columns=['ProdTaken'])
    y_test = data['ProdTaken']
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    results = {'accuracy': accuracy, 'f1_score': f1}
    with open(evaluation_output_path, 'w') as f:
        json.dump(results, f)
    print(f"Model Accuracy: {accuracy}")
    print(f"Model F1 Score: {f1}")

if __name__ == "__main__":
    evaluate_model()
