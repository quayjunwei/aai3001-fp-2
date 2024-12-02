import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

print("Current Python Path:", sys.path)

from src.etl.data_loading import (
    load_and_filter_csv,
    gather_image_paths,
    create_output_directory,
)

print(load_and_filter_csv)
