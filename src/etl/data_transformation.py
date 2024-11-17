import os
import shutil
from tqdm import tqdm


def create_output_dir(output_dir):
    """
    Create the output directory if it doesn't exist.

    Args:
        output_dir (str): Path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)


def sample_and_copy_images(data_filtered, allowed_labels, image_path_dict, output_dir, sample_sizes, default_sample_size):
    """
    Sample and copy images to class-specific folders.

    Args:
        data_filtered (pd.DataFrame): Filtered dataset.
        allowed_labels (list): List of allowed diagnosis labels.
        image_path_dict (dict): Dictionary of image filenames and paths.
        output_dir (str): Path to the output directory.
        sample_sizes (dict): Dictionary of specific sample sizes for certain labels.
        default_sample_size (int): Default sample size for unspecified labels.
    """
    for diagnosis in allowed_labels + ['Multiple Diagnosis']:
        # Filter data for the current diagnosis type
        if diagnosis == 'Multiple Diagnosis':
            subset = data_filtered[data_filtered['Diagnosis Type'] == diagnosis]
        else:
            subset = data_filtered[data_filtered['Finding Labels'] == diagnosis]

        # Determine the sample size
        sample_size = sample_sizes.get(diagnosis, default_sample_size)
        sample_size = min(sample_size, len(subset))  # Adjust for available data

        # Sample the data
        sampled_data = subset.sample(n=sample_size, random_state=42)

        # Create class-specific folder
        class_folder = os.path.join(output_dir, diagnosis.replace(' ', '_'))
        os.makedirs(class_folder, exist_ok=True)

        # Copy images to the folder
        for _, row in tqdm(sampled_data.iterrows(), desc=f"Copying images for {diagnosis}", total=len(sampled_data)):
            image_name = row['Image Index']
            image_path = image_path_dict.get(image_name)

            if image_path and os.path.exists(image_path):
                shutil.copy(image_path, class_folder)
            else:
                print(f"Warning: Image {image_name} not found in the directory structure.")

    print("Image extraction and organization complete.")
