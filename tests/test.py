import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Print sys.path to debug
print("Current Python Path:", sys.path)

from src.etl.data_loading import load_and_filter_csv, gather_image_paths, create_output_directory
print(load_and_filter_csv)
