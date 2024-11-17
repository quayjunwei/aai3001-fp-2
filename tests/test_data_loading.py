import os
import pandas as pd
import pytest
from src.etl.data_loading import load_and_filter_csv, gather_image_paths, create_output_directory

@pytest.fixture
def sample_csv(tmpdir):
    """
    Creates a temporary sample CSV file for testing.
    """
    csv_content = """Image Index,Finding Labels
    img1.png,Atelectasis|Cardiomegaly
    img2.png,Effusion
    img3.png,Pneumonia|No Finding
    img4.png,Unknown Label
    """
    csv_path = tmpdir.join("sample.csv")
    csv_path.write(csv_content)
    return str(csv_path)

@pytest.fixture
def allowed_labels():
    """
    Returns a list of allowed labels for testing.
    """
    return ["atelectasis", "cardiomegaly", "effusion", "pneumonia", "no finding"]

def test_load_and_filter_csv(sample_csv, allowed_labels):
    """
    Tests if load_and_filter_csv correctly filters the data.
    """
    data_filtered = load_and_filter_csv(sample_csv, allowed_labels)
    assert len(data_filtered) == 3  # Only 3 rows match the allowed labels
    assert "Unknown Label" not in data_filtered['Finding Labels'].to_list()

def test_gather_image_paths(tmpdir):
    """
    Tests if gather_image_paths correctly maps image filenames to their paths.
    """
    # Create mock image files
    image_dir = tmpdir.mkdir("images")
    img1 = image_dir.join("img1.png")
    img2 = image_dir.join("img2.png")
    img1.write("")
    img2.write("")
    
    image_dirs = str(image_dir)
    image_path_dict = gather_image_paths(image_dirs)
    
    assert len(image_path_dict) == 2
    assert "img1.png" in image_path_dict
    assert "img2.png" in image_path_dict

def test_create_output_directory(tmpdir):
    """
    Tests if create_output_directory creates the directory if it doesn't exist.
    """
    output_dir = tmpdir.join("output")
    assert not os.path.exists(output_dir)  # Directory shouldn't exist initially
    
    create_output_directory(str(output_dir))
    assert os.path.exists(output_dir)  # Directory should now exist
