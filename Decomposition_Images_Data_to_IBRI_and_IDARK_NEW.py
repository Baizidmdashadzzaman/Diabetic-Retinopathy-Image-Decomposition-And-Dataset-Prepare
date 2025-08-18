import os
import cv2
import numpy as np
from tqdm import tqdm  # For progress bar
# from PIL import Image  # Not strictly needed anymore as cv2 handles image loading/saving
# from skimage.morphology import disk, opening, closing, black_tophat  # Still useful, but less central

# --- Configuration ---
# IMPORTANT: Adjust these paths based on your local machine's file structure
# Input directory where your original fundus images are stored.
INPUT_IMAGES_DIR = 'Aptos_DDR_Dataset/Augmented/train'  # UPDATE THIS!

# Output directory for the new decomposed dataset.
OUTPUT_DATASET_DIR = 'Aptos_DDR_Dataset/Augmented/decomposed_fundus_dataset_new'  # UPDATED!

# Desired image size for processing. Ensure consistency.
IMAGE_SIZE = (512, 512)  # Updated to match your new script's size


# --- YOUR CUSTOM FUNCTIONS - UPDATED FOR NEW LESION DECOMPOSITION ---

def is_good_quality(image):
    """
    Checks if an image is of good quality based on simple intensity thresholds.
    This function now works with a NumPy array image directly, not a path.
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


def preprocess_image(image, target_size):
    """
    Applies initial preprocessing steps to the image (e.g., resizing).
    Normalization to [-1, 1] will now happen *after* lesion detection for saving.
    Assumes input `image` is a NumPy array (from cv2.imread).
    """
    # Resize using OpenCV, as it's consistent with cv2.imread and is faster
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    # No normalization here, as lesion detection works on [0, 255] or specific channels.
    return image_resized


def decompose_lesions_new(image):
    """
    Decomposes the image into bright and dark lesion maps using your NEW logic.
    Assumes input `image` is a BGR NumPy array in the [0, 255] range.
    It returns two single-channel maps, normalized to [-1, 1].
    """
    # --- Create a mask for the circular fundus ---
    gray_for_fundus_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, fundus_mask = cv2.threshold(gray_for_fundus_mask, 10, 255, cv2.THRESH_BINARY)
    kernel_fundus = np.ones((5, 5), np.uint8)
    fundus_mask = cv2.morphologyEx(fundus_mask, cv2.MORPH_CLOSE, kernel_fundus, iterations=2)
    fundus_mask = cv2.morphologyEx(fundus_mask, cv2.MORPH_OPEN, kernel_fundus, iterations=2)

    # --- Red Lesion Detection (Hemorrhages & Microaneurysms) ---
    green_channel = image[:, :, 1]
    clahe_red = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green = clahe_red.apply(green_channel)
    inverted = cv2.bitwise_not(enhanced_green)
    red_threshold = 200
    _, red_lesions = cv2.threshold(inverted, red_threshold, 255, cv2.THRESH_BINARY)
    kernel_red_base = np.ones((3, 3), np.uint8)
    red_lesions = cv2.morphologyEx(red_lesions, cv2.MORPH_OPEN, kernel_red_base, iterations=1)
    red_lesions = cv2.dilate(red_lesions, kernel_red_base, iterations=1)

    # --- APPLY FUNDUS MASK TO RED LESIONS ---
    red_lesions = cv2.bitwise_and(red_lesions, red_lesions, mask=fundus_mask)

    # --- Connected Components Analysis for Red Lesions ---
    num_labels_red, labels_red, stats_red, centroids_red = cv2.connectedComponentsWithStats(red_lesions, 8, cv2.CV_32S)
    final_red_lesions_filtered = np.zeros_like(red_lesions)
    min_area_red = 5
    max_area_red = 50000
    for i in range(1, num_labels_red):
        area = stats_red[i, cv2.CC_STAT_AREA]
        if area > min_area_red and area < max_area_red:
            final_red_lesions_filtered[labels_red == i] = 255
    red_lesions_map = final_red_lesions_filtered

    # --- Bright Lesion Detection (Hard Exudates) ---
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    clahe_bright = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(16, 16))
    enhanced_l_channel = clahe_bright.apply(l_channel)
    bright_threshold = 180
    _, bright_lesions = cv2.threshold(enhanced_l_channel, bright_threshold, 255, cv2.THRESH_BINARY)
    kernel_bright_open = np.ones((3, 3), np.uint8)
    kernel_bright_dilate = np.ones((5, 5), np.uint8)
    bright_lesions = cv2.morphologyEx(bright_lesions, cv2.MORPH_OPEN, kernel_bright_open, iterations=1)
    bright_lesions = cv2.dilate(bright_lesions, kernel_bright_dilate, iterations=1)

    # --- APPLY FUNDUS MASK TO BRIGHT LESIONS ---
    bright_lesions = cv2.bitwise_and(bright_lesions, bright_lesions, mask=fundus_mask)

    # Connected Components Analysis for bright lesions
    num_labels_bright, labels_bright, stats_bright, centroids_bright = cv2.connectedComponentsWithStats(bright_lesions, 8, cv2.CV_32S)
    final_bright_lesions_filtered = np.zeros_like(bright_lesions)
    for i in range(1, num_labels_bright):
        area = stats_bright[i, cv2.CC_STAT_AREA]
        if area > 12 and area < 3500:
            final_bright_lesions_filtered[labels_bright == i] = 255
    bright_lesions_map = final_bright_lesions_filtered

    # Normalize maps from [0, 255] to [-1, 1] for consistency with previous output format
    bright_map_norm = (bright_lesions_map / 127.5) - 1.0
    dark_map_norm = (red_lesions_map / 127.5) - 1.0 # Red lesions are dark lesions

    # Return the normalized maps, and the [0, 255] red and bright lesion masks for overlay
    return bright_map_norm, dark_map_norm, red_lesions_map, bright_lesions_map


# --- Main Processing Logic ---
def create_decomposed_dataset(input_dir, output_dir, image_size):
    """
    Processes images from input_dir, decomposes them, and saves
    original, bright, and dark maps to output_dir.
    Includes resume functionality by checking existing output files.
    Incorporates custom preprocessing and quality checks.
    Also creates an 'overlay' image combining original with lesion masks.
    """
    # Define subdirectories for the new dataset
    original_images_output_dir = os.path.join(output_dir, 'original_images')
    bright_maps_output_dir = os.path.join(output_dir, 'bright_maps')
    dark_maps_output_dir = os.path.join(output_dir, 'dark_maps') # This will now contain red lesions
    overlay_images_output_dir = os.path.join(output_dir, 'overlay_images') # NEW!
    low_quality_images_dir = os.path.join(output_dir, 'low_quality_images')

    # Create output directories if they don't exist
    os.makedirs(original_images_output_dir, exist_ok=True)
    os.makedirs(bright_maps_output_dir, exist_ok=True)
    os.makedirs(dark_maps_output_dir, exist_ok=True)
    os.makedirs(overlay_images_output_dir, exist_ok=True) # Create new overlay directory
    os.makedirs(low_quality_images_dir, exist_ok=True)

    print(f"Starting decomposition process...")
    print(f"Reading images from: {input_dir}")
    print(f"Saving decomposed data to: {output_dir}")

    # Get list of all image files (assuming common image extensions)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    if not image_files:
        print(f"No image files found in {input_dir}. Please check the path and file types.")
        return

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
        output_overlay_path = os.path.join(overlay_images_output_dir, filename.replace('.', '_overlay.')) # NEW!
        output_low_quality_path = os.path.join(low_quality_images_dir, filename)

        # Check for existing processed files (including the new overlay file)
        if (os.path.exists(output_original_path) and \
            os.path.exists(output_bright_path) and \
            os.path.exists(output_dark_path) and \
            os.path.exists(output_overlay_path)) or \
                os.path.exists(output_low_quality_path):
            skipped_files_count += 1
            continue

        try:
            image = cv2.imread(input_image_path)
            if image is None:
                print(f"Warning: Could not read image {filename}. Skipping.")
                failed_files.append(filename)
                continue

            # --- Apply Quality Check ---
            if not is_good_quality(image):
                print(f"Skipping {filename}: Marked as low quality.")
                cv2.imwrite(output_low_quality_path, image)
                low_quality_count += 1
                continue

            # --- Apply General Preprocessing (Resizing only now) ---
            preprocessed_img = preprocess_image(image, image_size) # preprocessed_img is now [0, 255] BGR

            # Generate bright and dark lesion maps using the NEW logic
            # The new decompose_lesions_new function expects [0, 255] BGR image
            bright_map_norm, dark_map_norm, red_lesions_mask, bright_lesions_mask = decompose_lesions_new(preprocessed_img)

            # Convert normalized maps from [-1, 1] back to [0, 255] for saving
            bright_map_save = ((bright_map_norm + 1.0) * 127.5).astype(np.uint8)
            dark_map_save = ((dark_map_norm + 1.0) * 127.5).astype(np.uint8)

            # --- Create the Overlay Image ---
            # Convert the preprocessed_img to RGB for overlay (as cv2.addWeighted works well with 3 channels)
            img_rgb_for_overlay = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB)

            # Create colored masks for overlay
            # Red lesions will be Red in the overlay
            red_mask_color = cv2.merge([red_lesions_mask, np.zeros_like(red_lesions_mask), np.zeros_like(red_lesions_mask)])
            # Bright lesions will be Cyan (Green + Blue) in the overlay
            bright_mask_color = cv2.merge([np.zeros_like(bright_lesions_mask), bright_lesions_mask, bright_lesions_mask])

            # Apply overlays
            overlay_image = cv2.addWeighted(img_rgb_for_overlay, 1.0, red_mask_color, 0.7, 0) # 0.7 opacity for red
            overlay_image = cv2.addWeighted(overlay_image, 1.0, bright_mask_color, 0.5, 0) # 0.5 opacity for bright

            # Save the images
            cv2.imwrite(output_original_path, preprocessed_img)
            cv2.imwrite(os.path.join(bright_maps_output_dir, filename.replace('.', '_bright.')), bright_map_save)
            cv2.imwrite(os.path.join(dark_maps_output_dir, filename.replace('.', '_dark.')), dark_map_save)
            cv2.imwrite(output_overlay_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)) # Convert back to BGR for saving

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