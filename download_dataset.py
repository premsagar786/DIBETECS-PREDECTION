import pandas as pd
import urllib.request

# Download the Pima Indians Diabetes Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Column names for the dataset
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

try:
    # Download and save the dataset
    print("Downloading diabetes dataset...")
    urllib.request.urlretrieve(url, 'data/diabetes.csv')
    
    # Read the CSV and add column names
    df = pd.read_csv('data/diabetes.csv', header=None, names=column_names)
    
    # Save with proper headers
    df.to_csv('data/diabetes.csv', index=False)
    
    print(f"✓ Dataset downloaded successfully!")
    print(f"✓ Total records: {len(df)}")
    print(f"✓ Features: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
except Exception as e:
    print(f"Error downloading dataset: {e}")
