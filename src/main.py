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

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ## Introduction
    This project aims to develop a deep learning model for automated diagnosis of 14 common thoracic diseases using chest X-ray images from the NIH Chest X-ray 14 dataset. Leveraging convolutional neural networks (CNNs) and extensive data augmentation, the model seeks to enhance diagnostic accuracy, reduce human error, and support medical professionals, especially in resource-limited settings. By rigorously training and fine-tuning the model, the project aspires to create a reliable tool for real-world clinical applications to improve healthcare quality.  

    [Github repo](https://github.com/quayjunwei/aai3001-fp-2)  
    [Data source](https://www.kaggle.com/datasets/nih-chest-xrays/data/data)
    """
)

# File uploader for X-ray image
img_data = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img_data is not None:
    st.image(img_data, caption="Uploaded image", use_column_width=True)

    # Load the pre-trained model
    @st.cache_resource
    def load_model():
        # Adjust path to model file
        model_path = os.path.join(
            os.path.dirname(__file__), "../models/densenet121_multilabel.pth"
        )
        model = models.densenet121(pretrained=False)
        model.classifier = torch.nn.Linear(
            model.classifier.in_features, 15
        )  # Adjust for 14 diseases

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

        # Adjust state_dict to avoid mismatches
        state_dict = checkpoint
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        model_state = model.state_dict()
        for key in state_dict.keys():
            if key in model_state and state_dict[key].shape == model_state[key].shape:
                model_state[key] = state_dict[key]

        model.load_state_dict(model_state)
        model.eval()  # Set model to evaluation mode
        return model

    model = load_model()

    # Preprocess the image
    def preprocess_image(image):
        preprocess = transforms.Compose(
            [
                transforms.Grayscale(
                    num_output_channels=3
                ),  # Ensure 3 channels for input
                transforms.Resize((224, 224)),  # Resize to model's expected input size
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # Normalize using ImageNet stats
            ]
        )
        return preprocess(image).unsqueeze(0)  # Add batch dimension

    # Predict probabilities
    def predict(image):
        preprocessed_img = preprocess_image(image)
        with torch.no_grad():
            outputs = model(preprocessed_img)
            probabilities = (
                torch.sigmoid(outputs).cpu().numpy()[0]
            )  # Use sigmoid for multi-label classification
        return probabilities

    # Process uploaded image
    image = Image.open(img_data).convert("RGB")
    probabilities = predict(image)

    # Display probabilities for each disease
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

    st.write("### Predicted Probabilities")
    for label, prob in zip(disease_labels, probabilities):
        st.write(f"{label}: {prob:.4f}")
