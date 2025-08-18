import pandas as pd
import os

def merge_csv_files(file_path1, file_path2, output_file_path, ignore_index=True):
    """
    Merges two CSV files into a single CSV file.

    Args:
        file_path1 (str): The path to the first CSV file.
        file_path2 (str): The path to the second CSV file.
        output_file_path (str): The path where the merged CSV file will be saved.
        ignore_index (bool): If True, the original index will be dropped and a new
                             one will be created for the merged DataFrame. Defaults to True.
    """
    if not os.path.exists(file_path1):
        print(f"Error: First CSV file not found at '{file_path1}'")
        return

    if not os.path.exists(file_path2):
        print(f"Error: Second CSV file not found at '{file_path2}'")
        return

    print(f"Reading '{file_path1}'...")
    try:
        df1 = pd.read_csv(file_path1)
        print(f"'{file_path1}' has {len(df1)} rows and {len(df1.columns)} columns.")
    except Exception as e:
        print(f"Error reading first CSV file '{file_path1}': {e}")
        return

    print(f"Reading '{file_path2}'...")
    try:
        df2 = pd.read_csv(file_path2)
        print(f"'{file_path2}' has {len(df2)} rows and {len(df2.columns)} columns.")
    except Exception as e:
        print(f"Error reading second CSV file '{file_path2}': {e}")
        return

    print("Merging CSV files...")
    try:
        # Concatenate the two DataFrames
        # pd.concat stacks them vertically.
        # If columns are not identical, it will fill missing values with NaN.
        merged_df = pd.concat([df1, df2], ignore_index=ignore_index)
        print(f"Merged DataFrame has {len(merged_df)} rows and {len(merged_df.columns)} columns.")
    except Exception as e:
        print(f"Error merging DataFrames: {e}")
        return

    print(f"Saving merged data to '{output_file_path}'...")
    try:
        merged_df.to_csv(output_file_path, index=False)
        print(f"Successfully merged files and saved to '{output_file_path}'")
    except Exception as e:
        print(f"Error saving merged CSV file: {e}")

# --- How to use this script ---
if __name__ == "__main__":
    # IMPORTANT: Set these paths to your actual file locations
    csv_file_1 = 'train_qa_prep_updated.csv'
    csv_file_2 = 'test_qa_prep_updated.csv'

    # The name for your combined CSV file
    output_combined_csv = 'combined_qa_prep.csv'

    merge_csv_files(
        file_path1=csv_file_1,
        file_path2=csv_file_2,
        output_file_path=output_combined_csv
    )

    print("\nScript finished. Check the output CSV file.")
    print("Remember to install pandas if you haven't: pip install pandas")

