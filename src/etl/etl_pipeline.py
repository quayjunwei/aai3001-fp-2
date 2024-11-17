from etl.data_loading import load_dataset, filter_data, load_image_paths
from etl.data_transformation import create_output_dir, sample_and_copy_images


def main():
    # Paths
    csv_path = "../../data/raw/Data_Entry_2017.csv"
    base_image_dir = "../../data/raw/images"
    output_dir = "../../data/processed"

    # Parameters
    allowed_labels = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
        "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
        "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"
    ]
    sample_sizes = {
        'No Finding': 2000,
        'Multiple Diagnosis': 4000
    }
    default_sample_size = 1000

    # Step 1: Load and filter the dataset
    print("Loading dataset...")
    data = load_dataset(csv_path)
    data_filtered = filter_data(data, allowed_labels)

    # Step 2: Load image paths
    print("Loading image paths...")
    image_path_dict = load_image_paths(base_image_dir)

    # Step 3: Create output directory
    print("Creating output directory...")
    create_output_dir(output_dir)

    # Step 4: Sample and copy images
    print("Sampling and copying images...")
    sample_and_copy_images(data_filtered, allowed_labels, image_path_dict, output_dir, sample_sizes, default_sample_size)

    print("ETL pipeline completed successfully.")


if __name__ == "__main__":
    main()
