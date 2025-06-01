import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
from skimage.transform import resize  # For resizing to 256×256
import gc  # For garbage collection

# Parameters
xyz_file = r""
output_dir = r""
low_quality_dir = os.path.join(output_dir, "low_quality")
grid_resolution = 0.001  # Resolution of the grid in degrees
chunk_size = 24  # Chunk size in pixels for UNet (before resizing)
final_image_size = (256, 256)  # Final image size for UNet
overlap = 0.1  # 10% overlap

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(low_quality_dir, exist_ok=True)

# Step 1: Read the full dataset
print("Reading the dataset...")
try:
    data = pd.read_csv(
        xyz_file,
        sep='\t',
        header=None,
        names=["longitude", "latitude", "depth"],
        encoding='latin1',
        low_memory=False,
        on_bad_lines='skip'
    )
except UnicodeDecodeError as e:
    print(f"Error reading the file due to encoding issue: {e}")
    raise

# Step 2: Clean the data
print("Cleaning the dataset...")
data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
data["depth"] = pd.to_numeric(data["depth"], errors="coerce")

# Drop rows with NaN values (caused by invalid data)
initial_row_count = len(data)
data = data.dropna(subset=["longitude", "latitude", "depth"])
final_row_count = len(data)
print(f"Removed {initial_row_count - final_row_count} invalid rows. Remaining rows: {final_row_count}")

# Compute global depth range for normalization
global_min = data["depth"].min()
global_max = data["depth"].max()
print(f"Using global depth range: {global_min} m … {global_max} m")

# Step 3: Determine global (spatial) boundaries
print("Determining global boundaries...")
x_min, x_max = data["longitude"].min(), data["longitude"].max()
y_min, y_max = data["latitude"].min(), data["latitude"].max()
print(f"Global boundaries: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

# Step 4: Calculate chunk dimensions with overlap
chunk_width = chunk_size * grid_resolution
chunk_height = chunk_size * grid_resolution
overlap_width = chunk_width * overlap
overlap_height = chunk_height * overlap

num_chunks_x = int(np.ceil((x_max - x_min) / (chunk_width - overlap_width)))
num_chunks_y = int(np.ceil((y_max - y_min) / (chunk_height - overlap_height)))

chunks = []
for i in range(num_chunks_x):
    for j in range(num_chunks_y):
        x_start = x_min + i * (chunk_width - overlap_width)
        x_end = x_start + chunk_width
        y_start = y_min + j * (chunk_height - overlap_height)
        y_end = y_start + chunk_height

        # Validate chunk boundaries
        if x_end > x_max or y_end > y_max:
            print(f"Skipping chunk with out-of-bound limits: x=[{x_start}, {x_end}], y=[{y_start}, {y_end}]")
            continue

        # Filter data for this chunk
        chunk = data[
            (data["longitude"] >= x_start) & (data["longitude"] < x_end) &
            (data["latitude"] >= y_start) & (data["latitude"] < y_end)
        ]

        # Ensure chunk is valid
        if chunk.empty:
            print(f"Skipping empty chunk: x=[{x_start}, {x_end}], y=[{y_start}, {y_end}]")
            continue

        chunks.append({
            "x_start": x_start, "x_end": x_end,
            "y_start": y_start, "y_end": y_end,
            "data": chunk
        })
        print(f"Chunk created with overlap: x=[{x_start}, {x_end}], y=[{y_start}, {y_end}], points={len(chunk)}")

print(f"Total chunks created: {len(chunks)}")

# Initialize counters
skipped_chunks = 0
saved_chunks = 0
low_quality_chunks = 0
valid_data_lost = 0
valid_data_saved = 0

# Helper: generate augmentations (only 90° and 180° rotations)
def generate_augmentations(image_array):
    """
    Given a 2D normalized array in [0,1], returns a list of augmented variants:
      1. original
      2. rotate 90°
      3. rotate 180°
    """
    aug_list = []
    aug_list.append(image_array)                    # original
    aug_list.append(np.rot90(image_array, k=1))     # rotate 90°
    aug_list.append(np.rot90(image_array, k=2))     # rotate 180°
    return aug_list

# Step 5: Process each chunk with interpolation, resizing, normalization, and augmentation
for idx, chunk_info in enumerate(chunks):
    print(f"Processing chunk {idx + 1}/{len(chunks)}...")
    x_start, x_end = chunk_info["x_start"], chunk_info["x_end"]
    y_start, y_end = chunk_info["y_start"], chunk_info["y_end"]
    chunk_data = chunk_info["data"]

    # Define grid for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_start, x_end, chunk_size),
        np.linspace(y_start, y_end, chunk_size)
    )

    # Interpolate using griddata
    points = chunk_data[["longitude", "latitude"]].values
    values = chunk_data["depth"].values

    # Check if there are enough points for interpolation
    if len(points) < 4:
        print(f"Skipping chunk {idx + 1}: Not enough points for interpolation ({len(points)} points).")
        skipped_chunks += 1
        valid_data_lost += len(points)
        continue

    try:
        interpolated_grid = griddata(points, values, (grid_x, grid_y), method='linear')
    except Exception as e:
        print(f"Skipping chunk {idx + 1}: Error during interpolation - {e}.")
        skipped_chunks += 1
        valid_data_lost += len(points)
        continue

    # Check if interpolated grid is valid
    if interpolated_grid is None or np.isnan(interpolated_grid).all():
        print(f"Skipping chunk {idx + 1}: Interpolated grid is invalid.")
        skipped_chunks += 1
        valid_data_lost += len(points)
        continue

    # Proceed with resizing, normalization, and saving
    valid_points = np.count_nonzero(~np.isnan(interpolated_grid))
    total_points = interpolated_grid.size
    valid_percentage = (valid_points / total_points) * 100

    # Replace NaNs with chunk-mean
    interpolated_grid = np.nan_to_num(interpolated_grid, nan=np.nanmean(values))

    # Resize to final_image_size
    resized_grid = resize(interpolated_grid, final_image_size, mode='constant', anti_aliasing=True)

    # Clip & normalize using global_min and global_max
    clipped = np.clip(resized_grid, global_min, global_max)
    normalized_grid = (clipped - global_min) / (global_max - global_min)

    # Generate augmentations (original, 90°, 180°)
    augmented_images = generate_augmentations(normalized_grid)

    # Save each augmented variant
    for aug_idx, aug_img in enumerate(augmented_images):
        if valid_percentage < 60:
            low_quality_chunks += 1
            filename = f"low_quality_chunk_{idx + 1:04d}_aug{aug_idx}.png"
            output_png = os.path.join(low_quality_dir, filename)
            if aug_idx == 0:
                print(f"Chunk {idx + 1} saved as low-quality: {output_png}")
        else:
            saved_chunks += 1
            filename = f"chunk_{idx + 1:04d}_aug{aug_idx}.png"
            output_png = os.path.join(output_dir, filename)
            if aug_idx == 0:
                print(f"Chunk {idx + 1} saved as high-quality: {output_png}")

        plt.imsave(output_png, aug_img, cmap="viridis")

    valid_data_saved += len(points)

    # Free memory after processing each chunk
    del interpolated_grid, resized_grid, normalized_grid, augmented_images
    gc.collect()

# Calculate and print summary
total_chunks = len(chunks)
skip_percentage = (skipped_chunks / total_chunks) * 100
save_percentage = (saved_chunks / total_chunks) * 100
low_quality_percentage = (low_quality_chunks / total_chunks) * 100

total_valid_data = valid_data_saved + valid_data_lost
valid_data_lost_percentage = (valid_data_lost / total_valid_data) * 100 if total_valid_data > 0 else 0
valid_data_saved_percentage = (valid_data_saved / total_valid_data) * 100 if total_valid_data > 0 else 0

print(f"All chunks processed successfully.")
print(f"Total chunks: {total_chunks}")
print(f"Skipped chunks: {skipped_chunks} ({skip_percentage:.2f}%)")
print(f"High-quality chunks (including aug.): {saved_chunks} ({save_percentage:.2f}%)")
print(f"Low-quality chunks (including aug.): {low_quality_chunks} ({low_quality_percentage:.2f}%)")
print(f"Total valid data points: {total_valid_data}")
print(f"Valid data lost: {valid_data_lost} ({valid_data_lost_percentage:.2f}%)")
print(f"Valid data saved: {valid_data_saved} ({valid_data_saved_percentage:.2f}%)")
