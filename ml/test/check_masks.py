
import os
import numpy as np

MASK_DIR = r"dataset_split/train/masks"

total_pixels = 0
oil_pixels = 0
tiles_with_oil = 0
total_tiles = 0

for file in os.listdir(MASK_DIR):

    if not file.endswith(".npy"):
        continue

    mask = np.load(
        os.path.join(MASK_DIR, file)
    )

    total_tiles += 1

    oil = np.sum(mask > 0)

    if oil > 0:
        tiles_with_oil += 1
        oil_pixels += oil

    total_pixels += mask.size

print("Total tiles:", total_tiles)
print("Tiles containing oil:", tiles_with_oil)

print(
    "Oil pixels:",
    oil_pixels
)

print(
    "Total pixels:",
    total_pixels
)

print(
    "Oil percentage:",
    (oil_pixels / total_pixels) * 100,
    "%"
)