import os
import rasterio
import numpy as np

# =========================
# SETTINGS
# =========================

IMAGE_DIR = r"images"
MASK_DIR = r"masks"

OUTPUT_IMAGE_DIR = r"dataset/images"
OUTPUT_MASK_DIR = r"dataset/masks"

TILE_SIZE = 256

MIN_DB = -30
MAX_DB = 5

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)


# =========================
# GET IMAGE FILES
# =========================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".tif")
]

print(f"Found {len(image_files)} images.")


total_tiles = 0


# =========================
# PROCESS EACH IMAGE
# =========================

for filename in sorted(image_files):

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    mask_path = os.path.join(
        MASK_DIR,
        filename
    )

    # Make sure corresponding mask exists
    if not os.path.exists(mask_path):
        print(f"Skipping {filename}: mask not found")
        continue

    scene_name = os.path.splitext(filename)[0]

    print(f"\nProcessing: {scene_name}")

    with rasterio.open(image_path) as image_src, \
         rasterio.open(mask_path) as mask_src:

        # Confirm dimensions match
        if (
            image_src.width != mask_src.width
            or image_src.height != mask_src.height
        ):
            print("Skipping: image and mask dimensions don't match")
            continue

        scene_tiles = 0

        # Process using small windows
        for y in range(
            0,
            image_src.height,
            TILE_SIZE
        ):

            for x in range(
                0,
                image_src.width,
                TILE_SIZE
            ):

                width = min(
                    TILE_SIZE,
                    image_src.width - x
                )

                height = min(
                    TILE_SIZE,
                    image_src.height - y
                )

                # Skip incomplete edge tiles
                if (
                    width != TILE_SIZE
                    or height != TILE_SIZE
                ):
                    continue

                window = rasterio.windows.Window(
                    x,
                    y,
                    TILE_SIZE,
                    TILE_SIZE
                )

                # =========================
                # READ IMAGE
                # =========================

                image = image_src.read(
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

                # Normalize 0–1
                image = (
                    image - MIN_DB
                ) / (
                    MAX_DB - MIN_DB
                )

                image = image.astype(
                    np.float32
                )

                # =========================
                # READ MASK
                # =========================

                mask = mask_src.read(
                    1,
                    window=window
                )

                # Convert mask to binary
                mask = (
                    mask > 0
                ).astype(
                    np.uint8
                )

                # =========================
                # SAVE
                # =========================

                tile_name = (
                    f"{scene_name}_"
                    f"{scene_tiles:05d}.npy"
                )

                image_output = os.path.join(
                    OUTPUT_IMAGE_DIR,
                    tile_name
                )

                mask_output = os.path.join(
                    OUTPUT_MASK_DIR,
                    tile_name
                )

                np.save(
                    image_output,
                    image
                )

                np.save(
                    mask_output,
                    mask
                )

                scene_tiles += 1
                total_tiles += 1

        print(
            f"Created {scene_tiles} tile pairs"
        )


# =========================
# COMPLETE
# =========================

print("\n==============================")
print("DATASET PREPARATION COMPLETE")
print("==============================")

print(
    f"Total image/mask tile pairs: {total_tiles}"
)

print(
    f"Images: {OUTPUT_IMAGE_DIR}"
)

print(
    f"Masks:  {OUTPUT_MASK_DIR}"
)