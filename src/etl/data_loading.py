import os
import pandas as pd
import glob

def load_and_filter_csv(csv_path, allowed_labels):
    """
    Loads a CSV file and filters rows based on allowed labels.
    
    Args:
        csv_path (str): Path to the CSV file.
        allowed_labels (list): List of allowed labels (case insensitive).
    
    Returns:
        pandas.DataFrame: A filtered DataFrame containing rows where any of the allowed labels are found in the 'Finding Labels' column.
    """
    data = pd.read_csv(csv_path)
    data_filtered = data[data['Finding Labels'].apply(
        lambda x: any(label in x.lower() for label in allowed_labels)
    )]
    return data_filtered

def gather_image_paths(image_dirs):
    """
    Gathers image paths from specified directories.
    
    Args:
        image_dirs (str): A glob pattern matching directories containing images.
    
    Returns:
        dict: A dictionary mapping image filenames to their absolute paths.
    """
    image_path_dict = {}
    for image_dir in glob.glob(image_dirs):
        for image_path in glob.glob(f"{image_dir}/*"):
            image_name = os.path.basename(image_path)
            image_path_dict[image_name] = image_path
    return image_path_dict

def create_output_directory(output_dir):
    """
    Creates the output directory if it doesn't already exist.
    
    Args:
        output_dir (str): Path to the output directory.
    
    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
