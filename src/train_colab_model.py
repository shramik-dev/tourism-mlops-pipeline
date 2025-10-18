
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

# Clean the data (same steps as in the EDA cell and data_prep)
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)
data_clean = data.dropna()
if 'CustomerID' in data_clean.columns: # Check if CustomerID exists before dropping
    data_clean = data_clean.drop(columns=['CustomerID'])
if 'Gender' in data_clean.columns: # Check if Gender exists before replacing
    data_clean['Gender'] = data_clean['Gender'].replace('Fe Male', 'Female')

# Define features and target
X = data_clean.drop('ProdTaken', axis=1)
y = data_clean['ProdTaken']

# Define preprocessing steps (consistent with train.py)
num_cols = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
            'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore',
            'NumberOfChildrenVisiting', 'MonthlyIncome']
cat_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
            'MaritalStatus', 'Designation', 'CityTier']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

# Create the full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(random_state=42))])

# Train the pipeline
pipeline.fit(X, y)

# Extract and save column names AFTER preprocessing
dummy_df = pd.DataFrame(columns=X.columns)
preprocessor.fit(dummy_df) # Fit preprocessor to get feature names
feature_names = []
for name, transformer, cols in preprocessor.transformers_:
    if hasattr(transformer, 'get_feature_names_out'):
        feature_names.extend(transformer.get_feature_names_out(cols))
    else:
        feature_names.extend(cols) # Fallback for transformers without get_feature_names_out

columns = feature_names

os.makedirs('/content/models', exist_ok=True)
joblib.dump(columns, '/content/models/columns.joblib')
joblib.dump(pipeline, '/content/models/best_rf_model.joblib')
logging.info("Model and columns saved to /content/models/")
