
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

# Load processed data
df = pd.read_csv('data/processed.csv')

# Debugging: Print available columns and file content
print(f"Loaded DataFrame columns: {df.columns.tolist()}")
with open('data/processed.csv', 'r') as f:
    print(f"File content preview: {f.read()[:100]}...")

# Check available columns and select 'feature' and 'target'
if 'feature' not in df.columns or 'target' not in df.columns:
    print(f"Error: Required columns 'feature' or 'target' not found. Available columns: {df.columns.tolist()}")
    exit(1)

X = df[['feature']]
y = df['target']

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.joblib')
print("Model training completed. Saved to models/model.joblib")
