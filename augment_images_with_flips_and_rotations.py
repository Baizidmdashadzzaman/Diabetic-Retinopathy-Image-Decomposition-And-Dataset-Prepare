import os
from PIL import Image
import random

def augment_images_with_flips_and_rotations(
    source_folder,
    output_folder,
    num_augmentations_per_image=2,
    rotation_range=(-50, 50) # New parameter for rotation range
):
    """
    Augments images in a source folder using random horizontal/vertical flips and random rotations.
    The original image is also copied to the output folder.

    Args:
        source_folder (str): Path to the folder containing original images.
        output_folder (str): Path to the folder where augmented images will be saved.
        num_augmentations_per_image (int): Number of augmented images to generate
                                           per original image (excluding the original itself).
                                           Each augmentation will be a random flip and rotation.
        rotation_range (tuple): A tuple (min_degree, max_degree) specifying the
                                range for random rotations.
    """
    if not os.path.isdir(source_folder):
        print(f"Error: Source folder '{source_folder}' not found.")
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"Ensured output folder exists: '{output_folder}'")

    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]

    if not image_files:
        print(f"No image files found in '{source_folder}'. Supported formats: .png, .jpg, .jpeg, .gif, .bmp, .webp")
        return

    print(f"Found {len(image_files)} images in '{source_folder}'.")
    total_generated_images = 0

    for i, filename in enumerate(image_files):
        original_path = os.path.join(source_folder, filename)
        name, ext = os.path.splitext(filename)

        try:
            img = Image.open(original_path)
            img = img.convert("RGB") # Ensure consistent mode for saving

            # 1. Save the original image
            original_output_path = os.path.join(output_folder, f"{name}_original{ext}")
            img.save(original_output_path)
            total_generated_images += 1

            # 2. Generate augmented images
            for aug_idx in range(num_augmentations_per_image):
                augmented_img = img.copy()

                # Apply random rotation
                angle = random.randint(rotation_range[0], rotation_range[1])
                augmented_img = augmented_img.rotate(angle, expand=False, fillcolor=(0,0,0)) # fillcolor for black background

                # Randomly apply horizontal flip
                if random.choice([True, False]):
                    augmented_img = augmented_img.transpose(Image.FLIP_LEFT_RIGHT)

                # Randomly apply vertical flip
                if random.choice([True, False]):
                    augmented_img = augmented_img.transpose(Image.FLIP_TOP_BOTTOM)

                # Ensure at least one transformation happened for the augmented versions
                # This is less critical with rotation, as rotation will almost always change the image
                # but it's good practice to ensure some change.
                # If the image somehow remains identical after random ops (highly unlikely with rotation)
                # we'll force a horizontal flip.
                if augmented_img == img:
                    augmented_img = augmented_img.transpose(Image.FLIP_LEFT_RIGHT)


                augmented_output_path = os.path.join(output_folder, f"{name}_aug{aug_idx+1}{ext}")
                augmented_img.save(augmented_output_path)
                total_generated_images += 1

            if (i + 1) % 100 == 0:
                print(f"Processed {i+1} images. Total generated: {total_generated_images}")

        except Exception as e:
            print(f"Error processing image '{filename}': {e}")

    print(f"\nAugmentation complete. Total original images processed: {len(image_files)}")
    print(f"Total augmented images generated (including originals): {total_generated_images}")
    print(f"Each original image generated {1 + num_augmentations_per_image} images.")

# --- How to use this script ---
if __name__ == "__main__":
    # IMPORTANT: Set your source image folder here
    source_images_folder = 'Aptos_DDR_Dataset/4' # Example: 'my_original_images'

    # IMPORTANT: Set your desired output folder for augmented images
    output_augmented_folder = 'Aptos_DDR_Dataset/4_augmented'

    # We want 3 images per original, so 1 original + 2 augmentations
    augment_images_with_flips_and_rotations(
        source_folder=source_images_folder,
        output_folder=output_augmented_folder,
        num_augmentations_per_image=2, # Generates 2 augmented images + 1 original = 3 total
        rotation_range=(-50, 50) # Random rotation between -50 and +50 degrees
    )

    print("\nScript finished. Check the output folder for augmented images.")


