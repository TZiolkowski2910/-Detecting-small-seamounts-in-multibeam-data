import os
import glob
import pandas as pd
import numpy as np
from pyproj import Transformer

def process_and_save_file(file, transformer, output_file):
    """Read a single file, validate, convert coordinates, and save."""
    # Read file with error handling
    try:
        data = pd.read_csv(
            file, header=None, names=["Easting", "Northing", "Depth"], sep=",",
            encoding='ISO-8859-1', on_bad_lines='skip'
        )
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        return 0

    # Check data types and remove invalid rows
    initial_rows = len(data)
    data = data[pd.to_numeric(data["Easting"], errors='coerce').notna()]
    data = data[pd.to_numeric(data["Northing"], errors='coerce').notna()]
    data = data[pd.to_numeric(data["Depth"], errors='coerce').notna()]
    cleaned_rows = len(data)

    removed_rows = initial_rows - cleaned_rows
    print(f"File {file}: removed {removed_rows} invalid rows.")

    if cleaned_rows == 0:
        return 0  # No valid data in the file

    # Convert UTM to Lon/Lat
    easting = data["Easting"].astype(float).values
    northing = data["Northing"].astype(float).values
    lon, lat = transformer.transform(easting, northing)
    data["Lon"] = lon
    data["Lat"] = lat

    # Keep only relevant columns
    data = data[["Lon", "Lat", "Depth"]]

    # Append results to the output file
    data.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False, float_format="%.6f")
    return cleaned_rows

if __name__ == '__main__':
    # Directory path to the input data
    data_folder = r""
    output_file = r""

    # Initialize the coordinate transformer
    transformer = Transformer.from_crs("epsg:32627", "epsg:4326", always_xy=True)

    # Get list of files
    file_list = glob.glob(os.path.join(data_folder, "*.txt"))

    # Initialize statistics
    total_files = len(file_list)
    total_valid_rows = 0

    print(f"Processing {total_files} files...")

    # Process each file individually
    for file_idx, file in enumerate(file_list, start=1):
        print(f"Processing file {file_idx}/{total_files}: {file}")
        valid_rows = process_and_save_file(file, transformer, output_file)
        total_valid_rows += valid_rows

    print(f"Processing complete. Results saved to {output_file}.")
    print(f"Total number of valid rows: {total_valid_rows}")
