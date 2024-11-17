import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

# Define the root directory of your dataset
data_dir = '/kaggle/working/extracted_classes'

# Define output directories for train and test folders
train_dir = '/kaggle/working/train'
test_dir = '/kaggle/working/test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load the full dataset with ImageFolder
full_dataset = ImageFolder(data_dir)

# Generate indices for train and test splits
train_indices, test_indices = train_test_split(
    list(range(len(full_dataset))), test_size=0.3, stratify=[label for _, label in full_dataset.samples], random_state=42
)

# Helper function to copy files to target folder
def copy_files(indices, target_dir):
    for idx in indices:
        file_path, class_idx = full_dataset.samples[idx]
        class_name = full_dataset.classes[class_idx]
        
        # Create the target directory for each class if it doesn't exist
        class_folder = os.path.join(target_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        # Copy the image to the appropriate class folder in the target directory
        shutil.copy(file_path, class_folder)

# Copy training images to the train directory
copy_files(train_indices, train_dir)
print(f"Training images saved in {train_dir}")

# Copy test images to the test directory
copy_files(test_indices, test_dir)
print(f"Test images saved in {test_dir}")
