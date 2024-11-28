import os
import pytest
from shutil import rmtree, copy
from sklearn.model_selection import train_test_split
from src.etl.data_splitting import main

@pytest.fixture
def setup_mock_dataset(tmpdir):
    """
    Sets up a mock dataset with class folders and images for testing.
    """
    dataset_dir = tmpdir.mkdir("extracted_classes")
    class_names = ["class1", "class2"]
    
    # Create mock class folders with dummy images
    for class_name in class_names:
        class_dir = dataset_dir.mkdir(class_name)
        for i in range(10):  # Add 10 mock images per class
            img = class_dir.join(f"image_{i}.png")
            img.write("")  # Create an empty file to represent an image
    
    # Create train, val, and test directories
    split_base_dir = tmpdir.mkdir("split")
    train_dir = split_base_dir.mkdir("train")
    val_dir = split_base_dir.mkdir("val")
    test_dir = split_base_dir.mkdir("test")
    
    return {
        "dataset_dir": str(dataset_dir),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "test_dir": str(test_dir),
    }

def test_data_splitting_ratios(setup_mock_dataset, monkeypatch):
    """
    Tests if the splitting ratios are correctly applied.
    """
    # Get the mock dataset directories
    dataset_dir = setup_mock_dataset["dataset_dir"]
    train_dir = setup_mock_dataset["train_dir"]
    val_dir = setup_mock_dataset["val_dir"]
    test_dir = setup_mock_dataset["test_dir"]
    
    # Monkeypatch the paths in data_splitting.py
    monkeypatch.setattr("src.etl.data_splitting.dataset_dir", dataset_dir)
    monkeypatch.setattr("src.etl.data_splitting.train_dir", train_dir)
    monkeypatch.setattr("src.etl.data_splitting.val_dir", val_dir)
    monkeypatch.setattr("src.etl.data_splitting.test_dir", test_dir)
    
    # Run the main function
    main()
    
    # Verify the number of images in each split for each class
    for split_name, split_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        for class_name in ["class1", "class2"]:
            class_split_dir = os.path.join(split_dir, class_name)
            assert os.path.exists(class_split_dir), f"{split_name} directory for {class_name} does not exist."
            num_images = len(os.listdir(class_split_dir))
            
            if split_name == "train":
                assert num_images == 7, f"{split_name} split has incorrect number of images."
            elif split_name == "val":
                assert num_images == 2, f"{split_name} split has incorrect number of images."
            elif split_name == "test":
                assert num_images == 1, f"{split_name} split has incorrect number of images."

def test_missing_directories_error(setup_mock_dataset, monkeypatch):
    """
    Tests if the script raises an error when required directories are missing.
    """
    # Get the mock dataset directory
    dataset_dir = setup_mock_dataset["dataset_dir"]
    train_dir = setup_mock_dataset["train_dir"]
    val_dir = setup_mock_dataset["val_dir"]
    test_dir = setup_mock_dataset["test_dir"]
    
    # Remove the train directory
    rmtree(train_dir)
    
    # Monkeypatch the paths in data_splitting.py
    monkeypatch.setattr("src.etl.data_splitting.dataset_dir", dataset_dir)
    monkeypatch.setattr("src.etl.data_splitting.train_dir", train_dir)
    monkeypatch.setattr("src.etl.data_splitting.val_dir", val_dir)
    monkeypatch.setattr("src.etl.data_splitting.test_dir", test_dir)
    
    # Expect the script to raise a FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Directory .* does not exist"):
        main()

def test_empty_class_directory(setup_mock_dataset, monkeypatch):
    """
    Tests if the script handles an empty class directory gracefully.
    """
    # Get the mock dataset directory
    dataset_dir = setup_mock_dataset["dataset_dir"]
    empty_class_dir = os.path.join(dataset_dir, "empty_class")
    os.makedirs(empty_class_dir)  # Create an empty class directory
    
    # Monkeypatch the paths in data_splitting.py
    monkeypatch.setattr("src.etl.data_splitting.dataset_dir", dataset_dir)
    monkeypatch.setattr("src.etl.data_splitting.train_dir", setup_mock_dataset["train_dir"])
    monkeypatch.setattr("src.etl.data_splitting.val_dir", setup_mock_dataset["val_dir"])
    monkeypatch.setattr("src.etl.data_splitting.test_dir", setup_mock_dataset["test_dir"])
    
    # Run the main function
    main()
    
    # Verify the empty class directory does not cause any errors
    for split_dir in [setup_mock_dataset["train_dir"], setup_mock_dataset["val_dir"], setup_mock_dataset["test_dir"]]:
        assert os.path.exists(split_dir), f"Split directory {split_dir} does not exist."
