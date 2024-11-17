import os
import shutil
from tqdm import tqdm
from PIL import Image
import random
from torchvision import transforms

def organize_images(data_filtered, image_path_dict, output_dir, allowed_labels, class_image_limit=1000):
    """
    Organizes images into class-specific folders, with an optional limit on the number of images per class.
    
    Args:
        data_filtered (pandas.DataFrame): Filtered DataFrame containing image metadata.
        image_path_dict (dict): Dictionary mapping image filenames to their paths.
        output_dir (str): Path to the directory where class-specific folders will be created.
        allowed_labels (list): List of allowed labels (case insensitive).
        class_image_limit (int): Maximum number of images to organize per class (default: 1000).
    
    Returns:
        dict: A dictionary with class labels as keys and the number of images organized for each class as values.
    """
    class_counts = {label: 0 for label in allowed_labels}
    
    for _, row in tqdm(data_filtered.iterrows(), desc="Organizing images", total=len(data_filtered)):
        image_name = row['Image Index']
        labels = row['Finding Labels'].split('|')
        for label in labels:
            label = label.strip().lower()
            if label in class_counts:
                class_folder = os.path.join(output_dir, label.replace(' ', '_'))
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)
                # Limit the number of images per class
                if class_counts[label] < class_image_limit:
                    src_path = image_path_dict.get(image_name)
                    if src_path and os.path.exists(src_path):
                        dst_path = os.path.join(class_folder, image_name)
                        if not os.path.exists(dst_path):
                            shutil.copy(src_path, dst_path)
                            class_counts[label] += 1
    return class_counts

def augment_hernia_class(hernia_folder, target_num_images=1000):
    """
    Performs data augmentation on images in the 'Hernia' class folder to ensure a minimum number of images.
    
    Args:
        hernia_folder (str): Path to the folder containing images for the 'Hernia' class.
        target_num_images (int): Target number of images in the folder after augmentation (default: 1000).
    
    Returns:
        None
    """
    augmentations = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])
    
    hernia_images = os.listdir(hernia_folder)
    num_existing_images = len(hernia_images)
    num_to_augment = target_num_images - num_existing_images

    if num_to_augment > 0:
        print(f"Augmenting {num_to_augment} images for 'Hernia'...")

        while len(os.listdir(hernia_folder)) < target_num_images:
            image_name = random.choice(hernia_images)
            image_path = os.path.join(hernia_folder, image_name)

            with Image.open(image_path) as img:
                img = img.convert("RGB")

                augmented_image = augmentations(img)

                augmented_image = transforms.ToPILImage()(augmented_image)

                new_image_name = f"aug_{len(os.listdir(hernia_folder))}_{image_name}"
                augmented_image.save(os.path.join(hernia_folder, new_image_name))

        print(f"Augmented 'Hernia' class to 1000 images.")
    else:
        print(f"No augmentation needed. Current count: {num_existing_images}")
