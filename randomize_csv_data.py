import pandas as pd
import os

def randomize_csv_data(input_csv_path, output_csv_path):
    """
    Reads a CSV file, randomizes the order of its rows, and saves the result
    to a new CSV file.

    Args:
        input_csv_path (str): The path to the input CSV file that needs to be randomized.
        output_csv_path (str): The path where the randomized CSV file will be saved.
    """
    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found at '{input_csv_path}'")
        return

    print(f"Reading CSV file from '{input_csv_path}'...")
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Original CSV has {len(df)} rows and {len(df.columns)} columns.")
    except Exception as e:
        print(f"Error reading input CSV file: {e}")
        return

    print("Randomizing the order of rows...")
    # Randomize the order of the rows in the DataFrame
    # frac=1 means return all rows in random order
    # reset_index(drop=True) creates a new default index
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    print(f"Saving randomized data to '{output_csv_path}'...")
    try:
        shuffled_df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully randomized CSV and saved to '{output_csv_path}'")
        print(f"Randomized CSV has {len(shuffled_df)} rows.")
    except Exception as e:
        print(f"Error saving randomized CSV file: {e}")

# --- How to use this script ---
if __name__ == "__main__":
    # IMPORTANT: Set the path to your input CSV file here
    your_input_csv = 'Aptos_DDR_Dataset/Augmented/train.csv' # Example: 'my_data.csv'

    # IMPORTANT: Set the desired name and path for your output randomized CSV file
    your_output_randomized_csv = 'Aptos_DDR_Dataset/Augmented/train_randomized.csv'

    randomize_csv_data(
        input_csv_path=your_input_csv,
        output_csv_path=your_output_randomized_csv
    )

    print("\nScript finished. Check the generated randomized CSV file.")
    print("Remember to install pandas if you haven't: pip install pandas")

