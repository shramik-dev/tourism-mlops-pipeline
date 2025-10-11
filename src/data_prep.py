
import os
import pandas as pd

# Dummy data preparation (replace with your actual logic)
data = {'feature': [1, 2, 3, 4], 'target': [0, 1, 0, 1]}  # Ensure 'feature' and 'target' are included
df = pd.DataFrame(data)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Save processed data and verify
df.to_csv('data/processed.csv', index=False)
print(f"Data preparation completed. Saved to data/processed.csv with columns: {df.columns.tolist()}")
with open('data/processed.csv', 'r') as f:
    print(f"File content preview: {f.read()[:100]}...")  # Print first 100 characters for debugging
