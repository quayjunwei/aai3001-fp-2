import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
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

img_data = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img_data is not None:
    st.image(img_data, caption="Uploaded image", use_column_width=True)

st.button("Scan Xray")