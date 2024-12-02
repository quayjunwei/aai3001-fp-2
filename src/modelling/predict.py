import os
import torch
from PIL import Image
from torchvision import transforms, models


def initialize_model(num_classes, model_load_path):
    """
    Initialize the DenseNet model with a custom classifier for prediction.

    Args:
        num_classes (int): Number of classes for classification.
        model_load_path (str): Path to the saved model checkpoint.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): Initialized model.
            - device (torch.device): Device where the model is loaded.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model, device


def predict_image(model, device, image_path, transform):
    """
    Predict probabilities for an input image using the trained model.

    Args:
        model (torch.nn.Module): Trained model.
        device (torch.device): Device where the model is loaded.
        image_path (str): Path to the input image.
        transform (callable): Transformations to apply to the image.

    Returns:
        numpy.ndarray: Predicted probabilities for each class.
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_probs = torch.sigmoid(outputs).cpu().squeeze().numpy()
    return predicted_probs


if __name__ == "__main__":
    # Define paths and labels
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    test_dir = os.path.join(base_dir, "data/processed/split/test")
    model_load_path = os.path.join(base_dir, "models/densenet121_epoch55.pth")
    class_labels = [
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

    # Initialize model
    model, device = initialize_model(len(class_labels), model_load_path)

    # Define transform
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Predict on a few test images
    image_paths = [os.path.join(test_dir, img) for img in os.listdir(test_dir)]
    for image_path in image_paths[:5]:
        probs = predict_image(model, device, image_path, transform)
        predictions = [class_labels[i] for i, p in enumerate(probs) if p > 0.5]
        print(f"Image: {image_path}, Predictions: {predictions}")
