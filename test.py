import pandas as pd

# Load the Parquet file
filepath = "Data/raw/test.parquet"  # Replace with your file path
try:
    df = pd.read_parquet(filepath)
    # Check if 'answers' column exists
    print(df['is_impossible'])
    if 'answers' in df.columns:
        # Check if any row has 'answers' set to True
        has_answers_true = df['is_impossible'].apply(lambda x: x == True).any()  # Or df['answers'] == True for direct boolean columns
        print(f"Any row with answers = True: {has_answers_true}")
    else:
        print("The 'answers' column is missing in the dataset.")
except Exception as e:
    print(f"Error reading the Parquet file: {e}")
