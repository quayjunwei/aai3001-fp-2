import os
import pandas as pd
import pytest
from src.etl.data_transformation import create_output_dir, sample_and_copy_images


# Test for create_output_dir
def test_create_output_dir(tmp_path):
    output_dir = tmp_path / "output"
    create_output_dir(output_dir)

    # Assertions
    assert output_dir.exists()
    assert output_dir.is_dir()


# Test for sample_and_copy_images
def test_sample_and_copy_images(tmp_path):
    # Create mock input data
    data = pd.DataFrame({
        "Image Index": ["img1.png", "img2.png", "img3.png"],
        "Finding Labels": ["Atelectasis", "No Finding", "Atelectasis"],
        "Diagnosis Type": ["Atelectasis", "No Finding", "Atelectasis"]
    })

    allowed_labels = ["Atelectasis", "No Finding"]
    image_path_dict = {
        "img1.png": str(tmp_path / "img1.png"),
        "img2.png": str(tmp_path / "img2.png"),
        "img3.png": str(tmp_path / "img3.png"),
    }

    # Create mock image files
    for img in image_path_dict.values():
        with open(img, "w") as f:
            f.write("mock data")

    # Define output directory
    output_dir = tmp_path / "processed"
    create_output_dir(output_dir)

    # Call sample_and_copy_images
    sample_and_copy_images(
        data_filtered=data,
        allowed_labels=allowed_labels,
        image_path_dict=image_path_dict,
        output_dir=output_dir,
        sample_sizes={"Atelectasis": 2, "No Finding": 1},
        default_sample_size=1,
    )

    # Assertions
    atelectasis_dir = output_dir / "Atelectasis"
    no_finding_dir = output_dir / "No_Finding"

    assert atelectasis_dir.exists()
    assert no_finding_dir.exists()
    assert len(list(atelectasis_dir.iterdir())) == 2
    assert len(list(no_finding_dir.iterdir())) == 1
