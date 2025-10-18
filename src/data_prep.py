
import pandas as pd
from datasets import load_dataset
import os

def prepare_data():
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
    if 'CustomerID' in data:
        data = data.drop('CustomerID', axis=1)
    if 'Gender' in data:
        data['Gender'] = data['Gender'].replace('Fe Male', 'Female')
    output_dir = os.getenv('OUTPUT_DIR', 'data')
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, 'processed.csv'), index=False)
    test_data = pd.DataFrame(dataset['test'])
    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop('Unnamed: 0', axis=1)
    test_data[num_cols] = test_data[num_cols].fillna(data[num_cols].median())
    test_data[cat_cols] = test_data[cat_cols].fillna('Unknown')
    if 'CustomerID' in test_data:
        test_data = test_data.drop('CustomerID', axis=1)
    if 'Gender' in test_data:
        test_data['Gender'] = test_data['Gender'].replace('Fe Male', 'Female')
    test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print("Data prepared and saved to", output_dir)

if __name__ == "__main__":
    prepare_data()
