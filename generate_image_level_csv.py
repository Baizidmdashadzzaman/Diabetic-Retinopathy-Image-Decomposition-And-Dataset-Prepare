import pandas as pd
import os

def generate_image_level_csv(base_folder_path, output_csv_path):
    """
    Scans subfolders within a base folder, extracts image filenames (without extension),
    and their parent folder name (level), then saves this data to a CSV file.

    Args:
        base_folder_path (str): The path to the base folder containing subfolders
                                (e.g., '0', '1', '2', '3', '4') with images.
        output_csv_path (str): The path where the generated CSV file will be saved.
    """
    if not os.path.isdir(base_folder_path):
        print(f"Error: Base folder '{base_folder_path}' not found.")
        return

    data = []
    # List of common image extensions to look for
    common_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')

    print(f"Scanning subfolders in '{base_folder_path}'...")

    # Walk through the base folder and its subdirectories
    for root, dirs, files in os.walk(base_folder_path):
        # The 'level' is the name of the direct parent folder
        # Skip the base folder itself if it contains files directly
        if root == base_folder_path:
            continue

        level = os.path.basename(root) # Get the current subfolder name (e.g., '0', '1')

        for filename in files:
            if filename.lower().endswith(common_extensions):
                # Extract filename without extension
                image_name_without_ext = os.path.splitext(filename)[0]
                data.append({'image': image_name_without_ext, 'level': level})

    if not data:
        print(f"No image files found in subfolders of '{base_folder_path}' with supported extensions.")
        return

    # Create a pandas DataFrame from the collected data
    df = pd.DataFrame(data)

    try:
        # Save the DataFrame to a CSV file
        df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully generated CSV file at '{output_csv_path}'")
        print(f"Total entries in CSV: {len(df)}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

# --- How to use this script ---
if __name__ == "__main__":
    # IMPORTANT: Set the path to your base folder containing the '0', '1', '2', '3', '4' subfolders
    your_base_image_directory = 'Aptos_DDR_Dataset/Augmented' # e.g., 'organized_images_by_class'

    # IMPORTANT: Set the desired name and path for your output CSV file
    output_csv_filename = 'Aptos_DDR_Dataset/Augmented/train.csv'

    generate_image_level_csv(
        base_folder_path=your_base_image_directory,
        output_csv_path=output_csv_filename
    )

    print("\nScript finished. Check the generated CSV file.")
    print("Remember to install pandas if you haven't: pip install pandas")

