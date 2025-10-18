
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datasets import load_dataset
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

dataset = load_dataset("Shramik121/tourism-split-dataset")
data = pd.DataFrame(dataset['train'])
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
os.makedirs('/content/models', exist_ok=True)
joblib.dump(columns, '/content/models/columns.joblib')
joblib.dump(pipeline, '/content/models/best_rf_model.joblib')
logging.info("Model and columns saved to /content/models/")
