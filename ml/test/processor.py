import os
import rasterio
import numpy as np

SENTINEL_DIR = r"masks"
OUTPUT_DIR = r"processed_tiles"

TILE_SIZE = 256
MIN_DB = -30
MAX_DB = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

tif_files = [
    f for f in os.listdir(SENTINEL_DIR)
    if f.lower().endswith((".tif", ".tiff"))
]

print(f"Found {len(tif_files)} TIFF files.")

total_tiles = 0

for file in tif_files:

    input_path = os.path.join(SENTINEL_DIR, file)

    # Remove extension
    scene_name = os.path.splitext(file)[0]

    scene_output = os.path.join(
        OUTPUT_DIR,
        scene_name
    )

    os.makedirs(scene_output, exist_ok=True)

    print(f"\nProcessing: {file}")

    with rasterio.open(input_path) as src:

        tile_number = 0

        # Move through image in 256x256 windows
        for y in range(0, src.height, TILE_SIZE):

            for x in range(0, src.width, TILE_SIZE):

                width = min(TILE_SIZE, src.width - x)
                height = min(TILE_SIZE, src.height - y)

                # Skip incomplete edge tiles
                if width != TILE_SIZE or height != TILE_SIZE:
                    continue

                window = rasterio.windows.Window(
                    x,
                    y,
                    TILE_SIZE,
                    TILE_SIZE
                )

                # Read only this small section
                image = src.read(
                    1,
                    window=window
                )

                # Replace invalid values
                image = np.nan_to_num(
                    image,
                    nan=MIN_DB,
                    posinf=MIN_DB,
                    neginf=MIN_DB
                )

                # Clip SAR values
                image = np.clip(
                    image,
                    MIN_DB,
                    MAX_DB
                )

                # Normalize to 0-1
                image = (
                    image - MIN_DB
                ) / (
                    MAX_DB - MIN_DB
                )

                image = image.astype(
                    np.float32
                )

                # Save tile
                output_file = os.path.join(
                    scene_output,
                    f"tile_{tile_number:05d}.npy"
                )

                np.save(
                    output_file,
                    image
                )

                tile_number += 1
                total_tiles += 1

        print(
            f"Created {tile_number} tiles from {file}"
        )


# =========================
# FINISHED
# =========================

print("\n==========================")
print("PREPROCESSING COMPLETE")
print("==========================")
print(f"Total TIFF files: {len(tif_files)}")
print(f"Total tiles: {total_tiles}")
print(f"Output folder: {OUTPUT_DIR}")