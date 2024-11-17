import os
import pandas as pd
import glob


def load_dataset(csv_path):
    """
    Load the CSV dataset.

    Args:
        csv_path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    return pd.read_csv(csv_path)


def filter_data(data, allowed_labels):
    """
    Filter rows to include only allowed labels and add a 'Diagnosis Type' column.

    Args:
        data (pd.DataFrame): Input dataset.
        allowed_labels (list): List of allowed diagnosis labels.

    Returns:
        pd.DataFrame: Filtered dataset.
    """
    data_filtered = data[data['Finding Labels'].apply(
        lambda x: any(label in x for label in allowed_labels)
    )]

    data_filtered['Diagnosis Type'] = data_filtered['Finding Labels'].apply(
        lambda x: 'Multiple Diagnosis' if '|' in x else x
    )

    return data_filtered


def load_image_paths(base_image_dir):
    """
    Get a dictionary of image paths based on the nested directory structure.

    Args:
        base_image_dir (str): Base directory where images are stored.

    Returns:
        dict: Dictionary with image filenames as keys and full paths as values.
    """
    all_image_paths = glob.glob(f"{base_image_dir}/**", recursive=True)
    return {os.path.basename(path): path for path in all_image_paths}
