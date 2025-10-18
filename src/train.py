
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from datasets import load_dataset

def train_model():
    # Load processed data
    data_path = os.getenv('DATA_PATH', 'data/processed.csv')
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        # Fallback to loading from dataset if file not found (e.g., in initial run)
        try:
            dataset = load_dataset("Shramik121/tourism-split-dataset")
            data = pd.DataFrame(dataset['train'])
            if 'Unnamed: 0' in data.columns:
                data = data.drop('Unnamed: 0', axis=1)
            data = data.dropna()
            if 'CustomerID' in data:
                data = data.drop('CustomerID', axis=1)
            if 'Gender' in data:
                data['Gender'] = data['Gender'].replace('Fe Male', 'Female')
            print("Loaded data from Hugging Face dataset.")
        except Exception as e:
            print(f"Failed to load data from file or Hugging Face: {e}")
            return # Exit if data cannot be loaded
    else:
        data = pd.read_csv(data_path)
        print(f"Loaded data from {data_path}")

    num_cols = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore',
                'NumberOfChildrenVisiting', 'MonthlyIncome']
    cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
                'MaritalStatus', 'Designation', 'CityTier']

    # Handle missing values (should be minimal after data_prep, but for robustness)
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
    dummy_processed = preprocessor.transform(dummy_df)

    # Get feature names from preprocessor
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols) # For transformers without get_feature_names_out

    columns = feature_names

    os.makedirs('models', exist_ok=True)
    joblib.dump(columns, 'models/columns.joblib')
    joblib.dump(pipeline, os.getenv('MODEL_OUTPUT', 'models/model.joblib'))
    print("Model and columns saved to models/")

if __name__ == "__main__":
    train_model()
