import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.morphology import disk, opening, closing, black_tophat
import imgaug.augmenters as iaa  # Import the augmentation library

# --- Configuration ---
# IMPORTANT: Adjust these paths based on your local machine's file structure
# Input directory where your original fundus images are stored.
INPUT_IMAGES_DIR = 'Aptos_DDR_Dataset/train'  # UPDATE THIS!

# Output directory for the new decomposed dataset.
OUTPUT_DATASET_DIR = 'Aptos_DDR_Dataset/decomposed_fundus_dataset'  # UPDATE THIS!

# Desired image size for processing. Ensure consistency.
IMAGE_SIZE = (640, 640)  # Width, Height (Matches your preferred size)

# --- Augmentation Configuration ---
# You can add or remove augmentations here to change the pipeline
# The number of augmented images to generate per original image
NUM_AUGMENTATIONS_PER_IMAGE = 3

# Define the sequence of augmentations to apply
AUGMENTATION_SEQUENCER = iaa.Sequential([
    # Add Gaussian blur with random sigma (0 to 1.5)
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.5))),
    # Apply affine transformations with a probability of 0.8
    iaa.Sometimes(0.8, iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        rotate=(-45, 45),
        shear=(-16, 16),
        order=[0, 1]  # Use nearest neighbor or bilinear interpolation
    )),
    # Change brightness and contrast with a probability of 0.5
    iaa.Sometimes(0.5, iaa.Multiply((0.7, 1.3), per_channel=0.2)),
    # Add Gaussian noise with a probability of 0.5
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))),
    # Randomly flip the images horizontally
    iaa.Fliplr(0.5),
    # Randomly flip the images vertically
    iaa.Flipud(0.5)
], random_order=True)


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
    low_quality_images_dir = os.path.join(output_dir, 'low_quality_images')

    # Create output directories if they don't exist
    os.makedirs(original_images_output_dir, exist_ok=True)
    os.makedirs(bright_maps_output_dir, exist_ok=True)
    os.makedirs(dark_maps_output_dir, exist_ok=True)
    os.makedirs(low_quality_images_dir, exist_ok=True)

    print(f"Starting initial decomposition process...")
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
        if (os.path.exists(output_original_path) and
            os.path.exists(output_bright_path) and
            os.path.exists(output_dark_path)) or \
                os.path.exists(output_low_quality_path):  # Check if already moved to low quality
            skipped_files_count += 1
            continue

        try:
            # Read image in BGR format
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

            # --- Apply General Preprocessing ---
            preprocessed_img = preprocess_image(image, image_size)

            # Generate bright and dark lesion maps using your refined decomposition logic
            bright_map, dark_map = decompose_lesions_original(preprocessed_img)

            # Convert normalized maps from [-1, 1] back to [0, 255] for saving
            bright_map_save = ((bright_map + 1.0) * 127.5).astype(np.uint8)
            dark_map_save = ((dark_map + 1.0) * 127.5).astype(np.uint8)

            # Save the images
            cv2.imwrite(output_original_path, ((preprocessed_img + 1.0) * 127.5).astype(np.uint8))
            cv2.imwrite(output_bright_path, bright_map_save)
            cv2.imwrite(output_dark_path, dark_map_save)

            processed_files_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed_files.append(filename)
            continue

    print(f"\nInitial decomposition complete.")
    print(f"Total images processed (newly): {processed_files_count}")
    print(f"Total images skipped (already existing): {skipped_files_count}")
    print(f"Total low quality images filtered: {low_quality_count}")
    if failed_files:
        print(f"Images that failed to process: {len(failed_files)}")
        with open(os.path.join(output_dir, 'failed_images.txt'), 'w') as f:
            for item in failed_files:
                f.write(f"{item}\n")
        print(f"A list of failed images has been saved to '{os.path.join(output_dir, 'failed_images.txt')}'")


def augment_and_decompose(input_dir, output_dir, image_size, num_augmentations, augmenter):
    """
    Reads from the pre-processed original images, applies augmentation,
    and then decomposes the augmented images into bright and dark maps.
    """
    print("\nStarting augmentation and decomposition process...")

    # Define new directories for augmented data
    original_augmented_dir = os.path.join(output_dir, 'original_augmented')
    bright_augmented_dir = os.path.join(output_dir, 'bright_augmented')
    dark_augmented_dir = os.path.join(output_dir, 'dark_augmented')

    # Create the new output directories
    os.makedirs(original_augmented_dir, exist_ok=True)
    os.makedirs(bright_augmented_dir, exist_ok=True)
    os.makedirs(dark_augmented_dir, exist_ok=True)

    # Get the list of all pre-processed original images to augment
    original_image_files = [f for f in os.listdir(input_dir) if
                            f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    if not original_image_files:
        print(f"No original pre-processed images found in {input_dir}. Please run the initial decomposition first.")
        return

    augmented_count = 0
    skipped_aug_count = 0
    failed_aug_files = []

    for filename in tqdm(original_image_files, desc="Augmenting and Decomposing"):
        input_image_path = os.path.join(input_dir, filename)

        try:
            # Read the pre-processed image in BGR format
            image_bgr = cv2.imread(input_image_path)
            if image_bgr is None:
                print(f"Warning: Could not read pre-processed image {filename}. Skipping.")
                failed_aug_files.append(filename)
                continue

            for i in range(num_augmentations):
                # Create a unique filename for the augmented image
                base_name, ext = os.path.splitext(filename)
                aug_filename = f"{base_name}_aug{i}{ext}"

                # Construct output file paths
                output_aug_original_path = os.path.join(original_augmented_dir, aug_filename)
                output_aug_bright_path = os.path.join(bright_augmented_dir, aug_filename.replace('.', '_bright.'))
                output_aug_dark_path = os.path.join(dark_augmented_dir, aug_filename.replace('.', '_dark.'))

                # Check if this specific augmented image and its maps already exist
                if (os.path.exists(output_aug_original_path) and
                        os.path.exists(output_aug_bright_path) and
                        os.path.exists(output_aug_dark_path)):
                    skipped_aug_count += 1
                    continue

                # Augment the image
                # imgaug expects RGB, so we convert. It also expects uint8, which cv2.imread provides.
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                augmented_image_rgb = augmenter.augment_image(image_rgb)

                # Convert augmented image back to BGR for OpenCV
                augmented_image_bgr = cv2.cvtColor(augmented_image_rgb, cv2.COLOR_RGB2BGR)

                # Normalize the augmented image for decomposition
                normalized_augmented_img = preprocess_image(augmented_image_bgr, image_size)

                # Decompose the augmented image
                bright_map, dark_map = decompose_lesions_original(normalized_augmented_img)

                # Convert normalized maps from [-1, 1] back to [0, 255] for saving
                bright_map_save = ((bright_map + 1.0) * 127.5).astype(np.uint8)
                dark_map_save = ((dark_map + 1.0) * 127.5).astype(np.uint8)

                # Save the augmented images
                cv2.imwrite(output_aug_original_path, augmented_image_bgr)
                cv2.imwrite(output_aug_bright_path, bright_map_save)
                cv2.imwrite(output_aug_dark_path, dark_map_save)

                augmented_count += 1

        except Exception as e:
            print(f"Error augmenting {filename}: {e}")
            failed_aug_files.append(filename)
            continue

    print(f"\nAugmentation and decomposition complete.")
    print(f"Total augmented images created: {augmented_count}")
    print(f"Total augmented images skipped (already existing): {skipped_aug_count}")
    print(f"Images that failed augmentation: {len(failed_aug_files)}")


# --- Run the pipeline ---
if __name__ == '__main__':
    # First, run the initial decomposition on the raw dataset
    create_decomposed_dataset(INPUT_IMAGES_DIR, OUTPUT_DATASET_DIR, IMAGE_SIZE)

    # Second, run the augmentation pipeline on the newly created pre-processed images
    original_images_output_dir = os.path.join(OUTPUT_DATASET_DIR, 'original_images')
    augment_and_decompose(original_images_output_dir, OUTPUT_DATASET_DIR, IMAGE_SIZE, NUM_AUGMENTATIONS_PER_IMAGE,
                          AUGMENTATION_SEQUENCER)
