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
import os
import shutil
import pytest
from PIL import Image
from src.etl.data_transformation import organize_images, augment_hernia_class

@pytest.fixture
def sample_filtered_data():
    """
    Creates a sample filtered DataFrame for testing.
    """
    import pandas as pd
    data = {
        "Image Index": ["img1.png", "img2.png", "img3.png"],
        "Finding Labels": ["atelectasis", "cardiomegaly|effusion", "pneumonia"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def image_path_dict(tmpdir):
    """
    Creates a temporary directory with mock images and returns their paths.
    """
    image_dir = tmpdir.mkdir("images")
    img1 = image_dir.join("img1.png")
    img2 = image_dir.join("img2.png")
    img3 = image_dir.join("img3.png")
    img1.write("")
    img2.write("")
    img3.write("")
    
    return {
        "img1.png": str(img1),
        "img2.png": str(img2),
        "img3.png": str(img3)
    }

def test_organize_images(sample_filtered_data, image_path_dict, tmpdir):
    """
    Tests if organize_images correctly moves images into class-specific folders.
    """
    output_dir = tmpdir.mkdir("output")
    allowed_labels = ["atelectasis", "cardiomegaly", "effusion", "pneumonia"]
    
    class_counts = organize_images(
        sample_filtered_data, image_path_dict, str(output_dir), allowed_labels
    )
    
    assert class_counts["atelectasis"] == 1
    assert class_counts["cardiomegaly"] == 1
    assert class_counts["effusion"] == 1
    assert class_counts["pneumonia"] == 1

    for label in allowed_labels:
        class_folder = os.path.join(output_dir, label)
        assert os.path.exists(class_folder)
        assert len(os.listdir(class_folder)) == 1

def test_augment_hernia_class(tmpdir):
    """
    Tests if augment_hernia_class correctly augments images in the 'hernia' folder.
    """
    hernia_folder = tmpdir.mkdir("hernia")
    img1 = hernia_folder.join("img1.png")
    img1.write("")  # Mock image file
    
    # Create a mock image
    with Image.new("RGB", (100, 100)) as img:
        img.save(str(img1))
    
    target_num_images = 5
    augment_hernia_class(str(hernia_folder), target_num_images)
    
    assert len(os.listdir(hernia_folder)) == target_num_images

    assert atelectasis_dir.exists()
    assert no_finding_dir.exists()
    assert len(list(atelectasis_dir.iterdir())) == 2
    assert len(list(no_finding_dir.iterdir())) == 1
