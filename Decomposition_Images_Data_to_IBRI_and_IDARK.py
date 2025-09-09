import os
import cv2
import numpy as np
from tqdm import tqdm  # For progress bar
from PIL import Image  # For is_good_quality and preprocess_image
from skimage.morphology import disk, opening, closing, black_tophat  # For decompose_lesions

# --- Configuration ---
# IMPORTANT: Adjust these paths based on your local machine's file structure
# Input directory where your original fundus images are stored.
INPUT_IMAGES_DIR = 'Aptos_DDR_Dataset/train'  # UPDATE THIS!

# Output directory for the new decomposed dataset.
OUTPUT_DATASET_DIR = 'Aptos_DDR_Dataset/decomposed_fundus_dataset'  # UPDATE THIS!

# Desired image size for processing. Ensure consistency.
IMAGE_SIZE = (640, 640)  # Width, Height (Matches your preferred size)


# --- YOUR CUSTOM FUNCTIONS - PASTED FROM diabetic-retinopathy-70-per-check-decomposition.ipynb ---

def is_good_quality(image):
    """
    Checks if an image is of good quality based on simple intensity thresholds.
    Adapted from your 'diabetic-retinopathy-70-per-check-decomposition.ipynb'
    This function will now work with a NumPy array image directly, not a path.
    """
    try:
        # Convert to grayscale if it's not already (for intensity checks)
        if len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image

        # Simple check: avoid totally black or white images
        if np.mean(img_gray) < 10 or np.mean(img_gray) > 245:
            return False
        return True
    except Exception as e:
        print(f"Error checking image quality: {e}")
        return False


def suppress_vessels(image, kernel_length):
    """
    Suppresses vessels in the retinal image using Gabor filters.
    Adapted from your 'diabetic-retinopathy-70-per-check-decomposition.ipynb'
    """
    if kernel_length % 2 == 0:
        kernel_length += 1
    angles = np.arange(0, 180, 15)
    vessel_response = np.zeros_like(image, dtype=np.float32)
    for angle in angles:
        kernel = cv2.getGaborKernel((kernel_length, kernel_length), sigma=kernel_length / 4.0,
                                    theta=np.deg2rad(angle), lambd=kernel_length / 2.0,
                                    gamma=0.5, psi=0)
        kernel -= kernel.mean()
        # Ensure image is float32 for filter2D
        filtered = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel)
        vessel_response = np.maximum(vessel_response, filtered)

    if vessel_response.max() > 0:
        vessel_response = (vessel_response - vessel_response.min()) / (vessel_response.max() - vessel_response.min())
    return vessel_response


def preprocess_image(image, target_size):
    """
    Applies initial preprocessing steps to the image (e.g., resizing, normalization).
    Adapted from your 'diabetic-retinopathy-70-per-check-decomposition.ipynb'
    Assumes input `image` is a NumPy array (from cv2.imread).
    """
    # Resize using OpenCV, as it's consistent with cv2.imread and is faster
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Normalize pixel values to [-1, 1] as per your notebook's `preprocess_image` in get_dataset
    img_array_normalized = (image_resized / 127.5) - 1.0

    return img_array_normalized


def decompose_lesions_original(image):
    """
    Decomposes the image into bright and dark lesion maps using your original logic.
    Assumes input `image` is already normalized to [-1, 1] and RGB (3 channels).
    This function replaces the previous get_bright_lesion_map and get_dark_lesion_map.
    It returns two single-channel maps normalized to [-1, 1].
    """
    # Ensure image is in range [0, 255] for OpenCV/skimage operations
    image_for_cv = ((image + 1.0) * 127.5).astype(np.uint8)

    # Use green channel for lesion detection as it generally provides best contrast for DR
    # Ensure image_for_cv is 3-channel for green channel extraction
    if image_for_cv.ndim == 3 and image_for_cv.shape[2] == 3:
        green_channel = image_for_cv[:, :, 1]
    else:
        # If it's grayscale or already 1-channel, use it directly
        green_channel = image_for_cv

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    green_proc = clahe.apply(green_channel)

    # Suppress vessels to isolate lesions - use green_proc as input for Gabor
    vessel_map = suppress_vessels(green_proc, 15)
    vessel_scaled = vessel_map * 0.8  # Scale down vessel contribution
    # Ensure (1 - vessel_scaled) is correctly handled for element-wise multiplication
    lesion_input = np.clip(green_proc * (1 - vessel_scaled), 0, 255).astype(
        np.uint8)  # Convert back to uint8 for skimage

    # --- Bright Lesion Detection (e.g., Exudates) ---
    se_bright = disk(7)  # Structuring element for bright lesions
    opened = opening(lesion_input, se_bright)  # Morphological opening
    bright_map = np.maximum(0, lesion_input.astype(np.float32) - opened.astype(np.float32))  # Calculate bright lesions
    bright_map = bright_map / bright_map.max() if bright_map.max() > 0 else bright_map  # Normalize to [0, 1]
    bright_map[bright_map < 0.015] = 0  # Thresholding to remove noise

    # --- Dark Lesion Detection (e.g., Hemorrhages, Microaneurysms) ---
    dark_maps = []
    for radius in [3, 5, 10, 15, 20]:  # Use multiple radii for different sizes of dark lesions
        se_dark = disk(radius)
        top_hat = black_tophat(lesion_input, se_dark).astype(np.float32)  # Morphological black top-hat
        if top_hat.max() > 0:
            top_hat /= top_hat.max()  # Normalize
        dark_maps.append(top_hat)
    dark_map = np.maximum.reduce(dark_maps)  # Combine maps from different radii
    dark_map[dark_map < 0.005] = 0  # Thresholding

    # Further refine dark map by closing small gaps
    closed = closing((dark_map > 0).astype(np.uint8) * 255, disk(5))
    dark_map = dark_map * (closed / 255.0)

    # Normalize bright_map and dark_map to [-1, 1] for model input
    bright_map_norm = (bright_map * 2.0) - 1.0  # From [0,1] to [-1,1]
    dark_map_norm = (dark_map * 2.0) - 1.0  # From [0,1] to [-1,1]

    return bright_map_norm, dark_map_norm


# --- Main Processing Logic ---
def create_decomposed_dataset(input_dir, output_dir, image_size):
    """
    Processes images from input_dir, decomposes them, and saves
    original, bright, and dark maps to output_dir.
    Includes resume functionality by checking existing output files.
    Incorporates custom preprocessing and quality checks.
    """
    # Define subdirectories for the new dataset
    original_images_output_dir = os.path.join(output_dir, 'original_images')
    bright_maps_output_dir = os.path.join(output_dir, 'bright_maps')
    dark_maps_output_dir = os.path.join(output_dir, 'dark_maps')
    low_quality_images_dir = os.path.join(output_dir, 'low_quality_images')  # New directory for rejected images

    # Create output directories if they don't exist
    os.makedirs(original_images_output_dir, exist_ok=True)
    os.makedirs(bright_maps_output_dir, exist_ok=True)
    os.makedirs(dark_maps_output_dir, exist_ok=True)
    os.makedirs(low_quality_images_dir, exist_ok=True)  # Create dir for low quality

    print(f"Starting decomposition process...")
    print(f"Reading images from: {input_dir}")
    print(f"Saving decomposed data to: {output_dir}")

    # Get list of all image files (assuming common image extensions)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    if not image_files:
        print(f"No image files found in {input_dir}. Please check the path and file types.")
        return

    # To keep track of processed files for logging
    processed_files_count = 0
    skipped_files_count = 0
    low_quality_count = 0
    failed_files = []

    # Process each image
    for filename in tqdm(image_files, desc="Processing images"):
        input_image_path = os.path.join(input_dir, filename)

        # Construct expected output file paths
        output_original_path = os.path.join(original_images_output_dir, filename)
        output_bright_path = os.path.join(bright_maps_output_dir, filename.replace('.', '_bright.'))
        output_dark_path = os.path.join(dark_maps_output_dir, filename.replace('.', '_dark.'))
        output_low_quality_path = os.path.join(low_quality_images_dir, filename)

        # Check if all three output files already exist, or if it was marked as low quality
        if (os.path.exists(output_original_path) and \
            os.path.exists(output_bright_path) and \
            os.path.exists(output_dark_path)) or \
                os.path.exists(output_low_quality_path):  # Check if already moved to low quality
            # print(f"Skipping {filename}: Already processed or marked as low quality.")
            skipped_files_count += 1
            continue  # Skip to the next image if already processed or flagged

        try:
            # Read image in BGR format
            image = cv2.imread(input_image_path)
            if image is None:
                print(f"Warning: Could not read image {filename}. Skipping.")
                failed_files.append(filename)
                continue

            # --- Apply Quality Check ---
            # Pass the loaded image array, not the path
            if not is_good_quality(image):
                print(f"Skipping {filename}: Marked as low quality.")
                cv2.imwrite(output_low_quality_path, image)  # Save low quality image to specific folder
                low_quality_count += 1
                continue

            # --- Apply General Preprocessing ---
            # This will resize and normalize to [-1, 1]
            preprocessed_img = preprocess_image(image, image_size)

            # Generate bright and dark lesion maps using your refined decomposition logic
            # decompose_lesions_original expects input in [-1, 1] and RGB (though it uses green channel)
            # OpenCV reads in BGR, so if your decompose_lesions logic relies on RGB,
            # you might need to convert `preprocessed_img` to RGB first:
            # preprocessed_img_rgb = cv2.cvtColor(((preprocessed_img + 1.0) * 127.5).astype(np.uint8), cv2.COLOR_BGR2RGB)
            # bright_map, dark_map = decompose_lesions_original((preprocessed_img_rgb / 127.5) - 1.0)

            # Assuming your decompose_lesions_original handles the color channels appropriately
            bright_map, dark_map = decompose_lesions_original(preprocessed_img)

            # Convert normalized maps from [-1, 1] back to [0, 255] for saving
            bright_map_save = ((bright_map + 1.0) * 127.5).astype(np.uint8)
            dark_map_save = ((dark_map + 1.0) * 127.5).astype(np.uint8)

            # Save the images
            # Original image (resized and preprocessed, convert back to [0, 255] for saving)
            cv2.imwrite(output_original_path, ((preprocessed_img + 1.0) * 127.5).astype(np.uint8))

            # Bright map (grayscale, so save as is)
            cv2.imwrite(os.path.join(bright_maps_output_dir, filename.replace('.', '_bright.')), bright_map_save)

            # Dark map (grayscale, so save as is)
            cv2.imwrite(os.path.join(dark_maps_output_dir, filename.replace('.', '_dark.')), dark_map_save)

            processed_files_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed_files.append(filename)
            continue

    print(f"\nDecomposition complete.")
    print(f"Total images processed (newly): {processed_files_count}")
    print(f"Total images skipped (already existing): {skipped_files_count}")
    print(f"Total low quality images filtered: {low_quality_count}")
    if failed_files:
        print(f"Images that failed to process: {len(failed_files)}")
        with open(os.path.join(output_dir, 'failed_images.txt'), 'w') as f:
            for item in failed_files:
                f.write(f"{item}\n")
        print(f"A list of failed images has been saved to '{os.path.join(output_dir, 'failed_images.txt')}'")


# --- Run the script ---
if __name__ == '__main__':
    create_decomposed_dataset(INPUT_IMAGES_DIR, OUTPUT_DATASET_DIR, IMAGE_SIZE)
