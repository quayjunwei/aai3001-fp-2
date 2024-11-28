import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def main():
    """
    Splits the processed dataset into train, validation, and test sets.
    Assumes the train, val, and test directories already exist.
    """
    # Define the main dataset directory (where images are stored by class)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Path to src/etl
    base_dir = os.path.abspath(os.path.join(script_dir, "../../"))  # Go up two directories
    
    dataset_dir = os.path.join(base_dir, "data/processed/extracted_classes")
    
    # Output directories for splits
    split_base_dir = os.path.join(base_dir, "data/processed/split")
    train_dir = os.path.join(split_base_dir, "train")
    val_dir = os.path.join(split_base_dir, "val")
    test_dir = os.path.join(split_base_dir, "test")
    
    # Define split ratios
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    
    # Iterate through each class folder
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        
        # Ensure it's a directory
        if os.path.isdir(class_path):
            # Get all image file names in this class
            images = os.listdir(class_path)
            
            # Split the data into train, val, and test sets
            train_images, temp_images = train_test_split(
                images, test_size=(val_ratio + test_ratio), random_state=42
            )
            val_images, test_images = train_test_split(
                temp_images, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42
            )
            
            # Create class-specific subdirectories in the splits
            for split_name, split_images in zip(
                ["train", "val", "test"], [train_images, val_images, test_images]
            ):
                split_class_dir = os.path.join(eval(f"{split_name}_dir"), class_name)
                
                if not os.path.exists(split_class_dir):
                    raise FileNotFoundError(
                        f"Directory '{split_class_dir}' does not exist. Please create it before running the script."
                    )
                
                # Copy images to the respective split folder
                for image_name in tqdm(
                    split_images, desc=f"Copying {split_name} images for class {class_name}"
                ):
                    src_path = os.path.join(class_path, image_name)
                    dst_path = os.path.join(split_class_dir, image_name)
                    shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    main()
