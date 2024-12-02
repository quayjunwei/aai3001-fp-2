import os
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import requests

# Streamlit page setup
st.set_page_config(
    page_title="JJASK Scan",
    page_icon="🫁",
)

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Home", "Model Prediction", "Pipeline"])

# Home Page
if page == "Home":
    st.title("Welcome to JJASK Scan")
    st.markdown(
        """
        ## About This Project
        Chest X-ray imaging is one of the most widely used and cost-effective tools for diagnosing various thoracic conditions. This project aims to develop a deep learning model for automated diagnosis of 14 common thoracic diseases using chest X-ray images from the NIH Chest X-ray 14 dataset.  
        - **Goal**: Enhance diagnostic accuracy and support medical professionals in resource-limited settings.
        - **Methods**: Leverages CNNs and extensive data augmentation.
        - **Outcome**: A reliable tool for real-world clinical applications to improve healthcare quality.

        [Github Repository](https://github.com/quayjunwei/aai3001-fp-2)

        ## [Data Source](https://www.kaggle.com/datasets/nih-chest-xrays/data/data): NIH Chest X-ray 14 Dataset
        The NIH Chest X-ray 14 dataset is one of the largest public datasets for medical image analysis, containing 112,120 frontal-view chest X-rays from 30,805 patients. It is annotated with 14 common thoracic diseases such as Atelectasis, Pneumonia, and Cardiomegaly, along with a "No Finding" label for normal cases.

        Key Features:
        - Multi-Label Annotations: X-rays can have multiple disease labels extracted via natural language processing of radiology reports.
        - Dataset Size: Large-scale with diverse patient demographics and imaging conditions.
        - Challenges: Includes class imbalance and potential label inaccuracies.
        - Applications: The dataset is widely used for developing and evaluating deep learning models for thoracic disease diagnosis and supports advancements in computer-aided diagnosis.

        """
    )

# Model Prediction Page
elif page == "Model Prediction":
    st.title("JJASK Scan - Model Prediction")

    # Model downloading and loading function
    def download_model(url, save_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    @st.cache_resource
    def load_model():
        model_path = "models/densenet121_epoch55.pth"
        if not os.path.exists(model_path):
            with st.spinner("Downloading model..."):
                model_url = "https://github.com/quayjunwei/aai3001-fp-2/releases/download/v1.0/densenet121_epoch55.pth"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                download_model(model_url, model_path)
        model = models.densenet121(pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 15)
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    model = load_model()

    # Preprocessing function
    def preprocess_image(image):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return transform(image).unsqueeze(0)

    # Prediction function
    def predict(image):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(input_tensor.to(torch.device("cpu")))
            probabilities = torch.sigmoid(outputs).cpu().squeeze().numpy()
        return probabilities

    # Define disease labels
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

    # File uploader for image
    img_data = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])
    if img_data is not None:
        st.image(img_data, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict Disease"):
            image = Image.open(img_data).convert("RGB")
            probabilities = predict(image)
            st.write("### Predicted Probabilities")
            for label, prob in zip(disease_labels, probabilities):
                st.write(f"{label}: {prob:.4f}")
            threshold = 0.5
            predicted_classes = [
                label
                for label, prob in zip(disease_labels, probabilities)
                if prob > threshold
            ]
            st.write("### Predicted Classes:")
            st.write(", ".join(predicted_classes) if predicted_classes else "None")


# Pipeline Tab
elif page == "Pipeline":
    st.title("Pipeline Overview")
    st.markdown(
        """
        ## Data Pipeline

        1. **Data Preprocessing**:
            - Images from the NIH Chest X-ray 14 dataset are filtered to retain only single-label cases.
            - Classes are balanced to ensure sufficient representation for each disease type (1000 images per class).

        2. **Data Augmentation**:
            - Techniques such as random rotation and cropping are applied to artificially increase the size and diversity of the dataset.

        3. **Train-Validation-Test Split**:
            - Dataset is split into training (70%), validation (20%), and test (10%) sets, ensuring balanced representation across classes.

        4. **Model Initialization**:
            - The base model is **DenseNet-121**, pre-trained on ImageNet.
            - Weights are further fine-tuned using **CheXNet pre-trained weights** ([GitHub Link](https://github.com/arnoweng/CheXNet)), which were specifically trained on chest X-ray images.

        5. **Fine-Tuning**:
            - The final classifier layer is replaced with a fully connected layer for 15 classes (14 diseases + "No Finding").
            - The last dense block of DenseNet-121 is unfrozen to allow fine-tuning for the specific task.

        6. **Training**:
            - **Loss Function**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss) for multi-label classification.
            - **Optimizer**: Adam optimizer with a learning rate of 0.0001.
            - Early stopping is implemented to prevent overfitting.

        7. **Evaluation**:
            - A threshold of 0.5 is used for classifying the presence of diseases.

        ## Pre-trained Weights: CheXNet
        - **CheXNet**: A DenseNet-121 model pre-trained on the NIH Chest X-ray 14 dataset for detecting thoracic diseases.
        - **Significance**: Leveraging CheXNet’s pre-trained weights accelerates training and improves model performance by transferring knowledge from a similar domain.
        - **Usage**:
            - The pre-trained weights are loaded and fine-tuned for multi-label classification specific to this project.
        """
    )
