import pandas as pd
import os
from rembg import remove # You need to install this library: pip install rembg
from PIL import Image    # rembg uses Pillow for image handling

def remove_background_from_images(
    source_image_folder,
    output_image_folder,
    csv_file_path=None, # Made optional
    image_column_name='image' # Column in CSV with image filenames (only used if csv_file_path is provided)
):
    """
    Removes backgrounds from images. Can either read image filenames from a CSV
    or process all common image files directly from a source folder.

    Args:
        source_image_folder (str): The path to the folder where the original images are located.
        output_image_folder (str): The path where the background-removed images will be saved.
        csv_file_path (str, optional): The path to the CSV file (e.g., 'combined_qa_prep.csv').
                                       If None, all images in source_image_folder will be processed.
        image_column_name (str): The name of the column in the CSV that contains
                                 the image file names (e.g., 'image'). Only used if csv_file_path is provided.
    """
    if not os.path.isdir(source_image_folder):
        print(f"Error: Source image folder not found at '{source_image_folder}'")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_image_folder, exist_ok=True)
    print(f"Ensured output folder exists: '{output_image_folder}'")

    image_filenames_to_process = []
    if csv_file_path:
        if not os.path.exists(csv_file_path):
            print(f"Error: CSV file not found at '{csv_file_path}'. Cannot process images based on CSV.")
            return

        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            print(f"Error reading CSV file '{csv_file_path}': {e}")
            return

        if image_column_name not in df.columns:
            print(f"Error: Image column '{image_column_name}' not found in the CSV file.")
            print(f"Available columns are: {df.columns.tolist()}")
            return

        image_filenames_to_process = df[image_column_name].astype(str).tolist()
        print(f"Processing {len(image_filenames_to_process)} image entries from '{csv_file_path}'...")
    else:
        # If no CSV, find all common image files in the source folder
        print(f"No CSV provided. Processing all common image files in '{source_image_folder}'...")
        common_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
        for filename in os.listdir(source_image_folder):
            if filename.lower().endswith(common_extensions):
                image_filenames_to_process.append(filename)
        print(f"Found {len(image_filenames_to_process)} images in '{source_image_folder}' to process.")

    if not image_filenames_to_process:
        print("No images found to process. Exiting.")
        return

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Iterate through each image filename
    for image_filename in image_filenames_to_process:
        source_image_path = os.path.join(source_image_folder, image_filename)

        # Construct the output path with a .png extension
        name, _ = os.path.splitext(image_filename)
        output_image_path = os.path.join(output_image_folder, f"{name}.png")

        if not os.path.exists(source_image_path):
            print(f"Warning: Source image not found, skipping: '{source_image_path}'")
            skipped_count += 1
            continue

        if os.path.exists(output_image_path):
            print(f"Info: Processed image already exists in destination, skipping: '{output_image_path}'")
            skipped_count += 1
            continue

        try:
            # Open the image
            input_image = Image.open(source_image_path)
            # Remove the background
            output_image = remove(input_image)
            # Save the processed image as PNG
            output_image.save(output_image_path)
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} images so far...")
        except Exception as e:
            print(f"Error processing '{source_image_path}': {e}")
            error_count += 1

    print("\n--- Background Removal Summary ---")
    print(f"Total image entries considered: {len(image_filenames_to_process)}")
    print(f"Images successfully processed: {processed_count}")
    print(f"Images skipped (not found at source or already processed): {skipped_count}")
    print(f"Errors encountered during processing: {error_count}")
    print(f"Background-removed images are saved in '{output_image_folder}'.")


# --- How to use this script ---
if __name__ == "__main__":
    # IMPORTANT: Set these paths to your actual file and folder locations

    # --- OPTION 1: Process images based on a CSV file ---
    # Uncomment and use this section if you want to process images specifically listed in a CSV.
    # csv_file_for_processing = 'combined_qa_prep.csv' # The CSV file containing image filenames
    # image_filename_column_in_csv = 'image' # The column in your CSV that contains the image filenames

    # source_folder_for_csv_images = 'train_qa_prep' # The folder where images referenced in the CSV are located.
    # output_folder_for_csv_processed_images = 'images_no_background_from_csv'

    # remove_background_from_images(
    #     csv_file_path=csv_file_for_processing,
    #     source_image_folder=source_folder_for_csv_images,
    #     output_image_folder=output_folder_for_csv_processed_images,
    #     image_column_name=image_filename_column_in_csv
    # )

    # --- OPTION 2: Process ALL common images in a folder (NO CSV needed) ---
    # Uncomment and use this section if you want to process all images directly from a folder.
    source_folder_all_images = 'combined_qa_prep/4' # The folder containing all images you want to process
    output_folder_for_all_processed_images = 'combined_qa_prep/4_transparent'

    remove_background_from_images(
        source_image_folder=source_folder_all_images,
        output_image_folder=output_folder_for_all_processed_images,
        csv_file_path=None # Set to None to process all images in the source folder
    )

    print("\nScript finished. Check the output folder(s) for images with backgrounds removed.")
    print("Remember to install necessary libraries: pip install pandas rembg Pillow")

