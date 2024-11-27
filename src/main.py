import os
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Streamlit page setup
st.set_page_config(
    page_title="JJASK Scan",
    page_icon="🫁",
)

st.write("# AAI3001 Final Project 2")

st.markdown(
    """
    ## Introduction
    This project aims to develop a deep learning model for automated diagnosis of 14 common thoracic diseases using chest X-ray images from the NIH Chest X-ray 14 dataset. Leveraging convolutional neural networks (CNNs) and extensive data augmentation, the model seeks to enhance diagnostic accuracy, reduce human error, and support medical professionals, especially in resource-limited settings. By rigorously training and fine-tuning the model, the project aspires to create a reliable tool for real-world clinical applications to improve healthcare quality.  

    [Github repo](https://github.com/quayjunwei/aai3001-fp-2)  
    [Data source](https://www.kaggle.com/datasets/nih-chest-xrays/data/data)
    """
)


# Load the pre-trained model
@st.cache_resource
def load_model():
    model_path = "models/densenet121_epoch55.pth"
    model = models.densenet121(pretrained=False)
    model.classifier = torch.nn.Linear(
        model.classifier.in_features, 15
    )  # Match the training structure

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set the model to evaluation mode
    return model


model = load_model()


# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to model's expected input size
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # Normalize using ImageNet stats
        ]
    )
    return transform(image).unsqueeze(0)  # Add batch dimension


# Define class labels
disease_labels = [
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
    "No Finding",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
]

# File uploader for a single X-ray image
img_data = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if img_data is not None:
    st.image(img_data, caption="Uploaded image", use_column_width=True)

    # Add a button to trigger prediction
    if st.button("Predict Disease"):
        # Process uploaded image
        image = Image.open(img_data).convert("RGB")

        # Predict probabilities
        def predict(image):
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                outputs = model(
                    input_tensor.to(torch.device("cpu"))
                )  # Ensure predictions run on the CPU
                probabilities = (
                    torch.sigmoid(outputs).cpu().squeeze().numpy()
                )  # Apply sigmoid for multi-label classification
            return probabilities

        probabilities = predict(image)

        # Display probabilities for each disease
        st.write("### Predicted Probabilities")
        for label, prob in zip(disease_labels, probabilities):
            st.write(f"{label}: {prob:.4f}")

        # Determine the predicted classes based on a threshold
        threshold = 0.5  # Adjust threshold as needed
        predicted_classes = [
            label
            for label, prob in zip(disease_labels, probabilities)
            if prob > threshold
        ]

        # Display predicted classes
        st.write("### Predicted Classes:")
        st.write(", ".join(predicted_classes) if predicted_classes else "None")
