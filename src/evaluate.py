
import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model():
    model = joblib.load('models/model.joblib')
    data = pd.DataFrame(load_dataset("Shramik121/tourism-split-dataset")['test'])
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    num_cols = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
                'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore', 
                'NumberOfChildrenVisiting', 'MonthlyIncome']
    cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
                'MaritalStatus', 'Designation', 'CityTier']
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    data[cat_cols] = data[cat_cols].fillna('Unknown')
    X = data.drop(columns=['ProdTaken'])
    y = data['ProdTaken']
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    logging.info(f"Model accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate_model()
