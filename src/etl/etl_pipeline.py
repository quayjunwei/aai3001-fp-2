import os
from data_loading import load_and_filter_csv, gather_image_paths, create_output_directory
from data_transformation import organize_images, augment_hernia_class

def main():
    """
    Orchestrates the entire ETL pipeline for image classification.
    
    Steps:
    1. Dynamically determines the base directory of the project.
    2. Defines paths for the raw data (CSV and images) and the output directory.
    3. Loads and filters data from the CSV file based on allowed labels.
    4. Gathers image paths from raw image directories.
    5. Organizes images into class-specific folders in the output directory.
    6. Performs data augmentation on the 'Hernia' class to ensure it has at least 1000 images.
    7. Displays the number of images per class.

    Returns:
        None
    """
    # Dynamically determine the base directory
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Path to src/etl
    base_dir = os.path.abspath(os.path.join(script_dir, "../../"))  # Go up two directories
    
    # Paths relative to the base directory
    csv_path = os.path.join(base_dir, "data/raw/Data_Entry_2017.csv")
    image_dirs = os.path.join(base_dir, "data/raw/images_*/images")  # Pattern to match all image directories
    output_dir = os.path.join(base_dir, "data/processed/extracted_classes")
    
    # Allowed labels
    allowed_labels = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
        "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
        "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"
    ]
    allowed_labels = [label.lower() for label in allowed_labels]  # Normalize for case insensitivity
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Load and filter data
    data_filtered = load_and_filter_csv(csv_path, allowed_labels)
    image_path_dict = gather_image_paths(image_dirs)
    
    # Organize images
    class_counts = organize_images(data_filtered, image_path_dict, output_dir, allowed_labels)
    
    # Display class counts
    print("\nNumber of images per class:")
    for label, count in class_counts.items():
        print(f"{label}: {count}")
    
    # Augment Hernia class
    hernia_folder = os.path.join(output_dir, "hernia")
    augment_hernia_class(hernia_folder)

if __name__ == "__main__":
    main()
