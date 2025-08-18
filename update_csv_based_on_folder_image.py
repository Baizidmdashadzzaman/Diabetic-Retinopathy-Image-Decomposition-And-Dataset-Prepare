import pandas as pd
import os

def validate_images_in_csv(csv_file_path, image_folder_path, image_column_name='image_path', output_csv_path=None):
    """
    Validates image paths in a CSV file against an image folder and creates a new CSV
    with only the valid entries. It also handles cases where image names in the CSV
    do not include file extensions by trying common extensions.

    Args:
        csv_file_path (str): The path to the input CSV file.
        image_folder_path (str): The path to the folder containing the images.
        image_column_name (str): The name of the column in the CSV that contains
                                 the image file names or relative paths.
                                 Defaults to 'image_path'.
        output_csv_path (str, optional): The path for the updated CSV file.
                                         If None, a default name like 'updated_yourfile.csv'
                                         will be used in the same directory as the input CSV.
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at '{csv_file_path}'")
        return

    if not os.path.isdir(image_folder_path):
        print(f"Error: Image folder not found at '{image_folder_path}'")
        return

    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if image_column_name not in df.columns:
        print(f"Error: Column '{image_column_name}' not found in the CSV file.")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    print(f"Original CSV has {len(df)} rows.")

    # List of common image extensions to try if none is provided in the CSV
    common_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    # Create a list to store rows with existing images
    valid_rows = []
    checked_count = 0
    found_count = 0

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        image_name_from_csv = str(row[image_column_name])
        found_image_path = None

        checked_count += 1
        if checked_count % 100 == 0:
            print(f"Processed {checked_count} images...")

        # Check if the image name from CSV already has an extension
        if os.path.splitext(image_name_from_csv)[1]: # If there's an extension
            full_image_path = os.path.join(image_folder_path, image_name_from_csv)
            if os.path.exists(full_image_path):
                found_image_path = image_name_from_csv
        else: # No extension in the CSV entry, try common ones
            for ext in common_extensions:
                potential_image_name = image_name_from_csv + ext
                full_image_path = os.path.join(image_folder_path, potential_image_name)
                if os.path.exists(full_image_path):
                    found_image_path = potential_image_name
                    break # Found it, no need to check other extensions

        if found_image_path:
            # Create a copy of the row and update the image_column_name with the full filename
            # This ensures the output CSV has the correct filename with extension
            new_row = row.copy()
            new_row[image_column_name] = found_image_path
            valid_rows.append(new_row)
            found_count += 1
        # else:
        #     print(f"Image not found for base name: {image_name_from_csv}") # Uncomment to see missing images

    # Create a new DataFrame from the valid rows
    updated_df = pd.DataFrame(valid_rows)

    # Determine the output CSV path
    if output_csv_path is None:
        csv_dir = os.path.dirname(csv_file_path)
        csv_name = os.path.basename(csv_file_path)
        name, ext = os.path.splitext(csv_name)
        output_csv_path = os.path.join(csv_dir, f"{name}_updated{ext}")

    try:
        # Save the updated DataFrame to a new CSV file
        updated_df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully created updated CSV file at '{output_csv_path}'")
        print(f"Original rows: {len(df)}")
        print(f"Images checked: {checked_count}")
        print(f"Valid images found: {found_count}")
        print(f"Rows removed (images not found): {len(df) - found_count}")
    except Exception as e:
        print(f"Error saving updated CSV file: {e}")

# --- How to use this script ---
if __name__ == "__main__":
    # IMPORTANT: Replace these paths with your actual file and folder paths
    your_csv_file = 'test_qa_prep.csv'
    your_image_folder = 'test_qa_prep'

    # IMPORTANT: Set this to the actual column name in your CSV that contains the image names
    image_column = 'image' # Changed to 'image' as per your input

    # Optional: Specify an output path, otherwise it will create 'train_qa_prep_updated.csv'
    # output_csv = 'cleaned_image_data.csv'

    validate_images_in_csv(
        csv_file_path=your_csv_file,
        image_folder_path=your_image_folder,
        image_column_name=image_column,
        # output_csv_path=output_csv # Uncomment this line to use a custom output path
    )

    print("\nScript finished. Check the output CSV file.")
    print("Remember to install pandas if you haven't: pip install pandas")

