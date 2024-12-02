import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image

# Define allowed labels and label mapping
allowed_labels = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No_Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]
label_to_index = {label.lower(): idx for idx, label in enumerate(allowed_labels)}


def load_dataset_paths_and_labels(data_dir, label_to_index):
    """
    Load dataset paths and corresponding labels.

    Args:
        data_dir (str): Path to the dataset directory.
        label_to_index (dict): Dictionary mapping class names to indices.

    Returns:
        tuple: A tuple containing:
            - image_paths (list): List of image file paths.
            - labels (list): List of multi-hot encoded labels for each image.
    """
    image_paths = []
    labels = []

    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, image_name))

                # Create a multi-hot encoded label
                multi_hot_label = torch.zeros(len(label_to_index), dtype=torch.float32)
                multi_hot_label[label_to_index[label.lower()]] = 1.0
                labels.append(multi_hot_label)

    return image_paths, labels


def initialize_model(num_classes, checkpoint_path):
    """
    Initialize the DenseNet model with a custom classifier.

    Args:
        num_classes (int): Number of classes for classification.
        checkpoint_path (str): Path to the model checkpoint file.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): Initialized model.
            - device (torch.device): Device where the model is loaded.
    """
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        print(f"No checkpoint found at {checkpoint_path}. Training from scratch.")

    return model, device


def train_model(
    model,
    device,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    model_save_path,
):
    """
    Train the DenseNet model.

    Args:
        model (torch.nn.Module): The model to train.
        device (torch.device): The device for training.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of training epochs.
        model_save_path (str): Path to save the trained model.

    Returns:
        None
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {running_loss / len(train_loader)}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    train_dir = os.path.join(base_dir, "data/processed/split/train")
    val_dir = os.path.join(base_dir, "data/processed/split/val")
    checkpoint_path = os.path.join(base_dir, "models/model.pth.tar")
    model_save_path = os.path.join(base_dir, "models/densenet121_epoch55.pth")

    # Load dataset paths and labels
    train_image_paths, train_labels = load_dataset_paths_and_labels(
        train_dir, label_to_index
    )
    val_image_paths, val_labels = load_dataset_paths_and_labels(val_dir, label_to_index)

    # Custom dataset
    class ChestXRayDataset(Dataset):
        """
        A custom PyTorch Dataset class for Chest X-Ray images.

        Args:
            image_paths (list): List of image file paths.
            labels (list): List of corresponding labels.
            transform (callable, optional): Transformations to apply to the images.

        Attributes:
            image_paths (list): List of image file paths.
            labels (list): List of corresponding labels.
            transform (callable, optional): Transformations to apply to the images.
        """

        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    # Define transforms
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Initialize model
    model, device = initialize_model(len(allowed_labels), checkpoint_path)

    # Initialize datasets and dataloaders
    train_dataset = ChestXRayDataset(
        train_image_paths, train_labels, transform=data_transforms["train"]
    )
    val_dataset = ChestXRayDataset(
        val_image_paths, val_labels, transform=data_transforms["val"]
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Define criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    train_model(
        model,
        device,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=60,
        model_save_path=model_save_path,
    )
