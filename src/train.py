
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from datasets import load_dataset
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
                       'Designation', 'CityTier', 'ProdTaken'] # Include target for validation
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
        ],
        remainder='passthrough' # Keep other columns (like Passport, OwnCar)
        )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

    pipeline.fit(X, y)

    # Extract and save the list of columns *after* preprocessing
    # This is crucial for the prediction script
    # We can create a dummy dataframe processed by the preprocessor to get column names
    dummy_df = pd.DataFrame(columns=X.columns)
    preprocessor.fit(dummy_df) # Fit preprocessor to get feature names
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols) # Fallback for transformers without get_feature_names_out


    columns = feature_names

    os.makedirs('models', exist_ok=True)
    joblib.dump(columns, 'models/columns.joblib')
    joblib.dump(pipeline, os.getenv('MODEL_OUTPUT', 'models/model.joblib'))
    logging.info("Model and columns saved to models/")

if __name__ == "__main__":
    train_model()
