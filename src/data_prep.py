
import pandas as pd
from datasets import load_dataset
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data():
    dataset = load_dataset("Shramik121/tourism-split-dataset")
    data = pd.DataFrame(dataset['train'])
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    data = data.dropna()
    if 'CustomerID' in data:
        data = data.drop('CustomerID', axis=1)
    if 'Gender' in data:
        data['Gender'] = data['Gender'].replace('Fe Male', 'Female')
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/processed.csv', index=False)
    data.to_csv('data/test.csv', index=False)
    logging.info("Data prepared and saved to data/processed.csv and data/test.csv")

if __name__ == "__main__":
    prepare_data()
