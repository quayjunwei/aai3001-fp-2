import os
import pandas as pd
import pytest
from src.etl.data_loading import load_dataset, filter_data, load_image_paths


# Test for load_dataset
def test_load_dataset(tmp_path):
    # Create a temporary CSV file
    csv_path = tmp_path / "test_data.csv"
    data = pd.DataFrame({
        "Image Index": ["img1.png", "img2.png"],
        "Finding Labels": ["Atelectasis", "No Finding"]
    })
    data.to_csv(csv_path, index=False)

    # Load the dataset
    loaded_data = load_dataset(csv_path)
    assert len(loaded_data) == 2
    assert list(loaded_data.columns) == ["Image Index", "Finding Labels"]


# Test for filter_data
def test_filter_data():
    # Create sample data
    data = pd.DataFrame({
        "Image Index": ["img1.png", "img2.png", "img3.png"],
        "Finding Labels": ["Atelectasis", "No Finding", "Atelectasis|Effusion"]
    })
    allowed_labels = ["Atelectasis", "No Finding"]

    # Filter data
    filtered_data = filter_data(data, allowed_labels)

    # Assertions
    assert len(filtered_data) == 3
    assert "Diagnosis Type" in filtered_data.columns
    assert filtered_data.iloc[2]["Diagnosis Type"] == "Multiple Diagnosis"


# Test for load_image_paths
def test_load_image_paths(tmp_path):
    # Create mock directory structure
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "img1.png").write_text("mock data")
    (img_dir / "img2.png").write_text("mock data")

    # Load image paths
    image_paths = load_image_paths(img_dir)

    # Assertions
    assert len(image_paths) == 2
    assert "img1.png" in image_paths
    assert "img2.png" in image_paths
