
import pandas as pd
from datasets import load_dataset
import os

def prepare_data():
    dataset = load_dataset("Shramik121/tourism-split-dataset")
    data = pd.DataFrame(dataset['train'])
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    # Drop rows with missing values for simplicity as done in EDA
    data = data.dropna()
    if 'CustomerID' in data:
        data = data.drop('CustomerID', axis=1)
    if 'Gender' in data:
        data['Gender'] = data['Gender'].replace('Fe Male', 'Female')

    # Save processed data to a specific location within the repo
    output_dir = os.getenv('OUTPUT_DIR', 'data')
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, 'processed.csv'), index=False)

    # Save test data to a specific location within the repo
    test_data = pd.DataFrame(dataset['test'])
    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop('Unnamed: 0', axis=1)
    test_data = test_data.dropna() # Also drop missing for consistency with train
    if 'CustomerID' in test_data:
        test_data = test_data.drop('CustomerID', axis=1)
    if 'Gender' in test_data:
        test_data['Gender'] = test_data['Gender'].replace('Fe Male', 'Female')

    test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print("Data prepared and saved to", output_dir)

if __name__ == "__main__":
    prepare_data()
