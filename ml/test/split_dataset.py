import os
import shutil
import random

IMAGE_DIR = r"dataset/images"
MASK_DIR = r"dataset/masks"

OUTPUT_DIR = r"dataset_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# -------------------------
# Get scenes
# -------------------------

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".npy")
]

scenes = set()

for file in image_files:
    # Everything before the final "_00000.npy"-style tile number
    scene = file.rsplit("_", 1)[0]
    scenes.add(scene)

scenes = sorted(scenes)

print("Scenes found:", len(scenes))
print(scenes)

# -------------------------
# Shuffle scenes
# -------------------------

random.shuffle(scenes)

total = len(scenes)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_scenes = scenes[:train_end]
val_scenes = scenes[train_end:val_end]
test_scenes = scenes[val_end:]

print("\nTRAIN:", train_scenes)
print("VALIDATION:", val_scenes)
print("TEST:", test_scenes)

# -------------------------
# Create folders
# -------------------------

for split in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(OUTPUT_DIR, split, "images"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(OUTPUT_DIR, split, "masks"),
        exist_ok=True
    )

# -------------------------
# Copy matching pairs
# -------------------------

def copy_scene(scene, split):

    count = 0

    for file in image_files:

        if not file.startswith(scene + "_"):
            continue

        image_file = os.path.join(
            IMAGE_DIR,
            file
        )

        mask_file = os.path.join(
            MASK_DIR,
            file
        )

        if not os.path.exists(mask_file):
            continue

        shutil.copy2(
            image_file,
            os.path.join(
                OUTPUT_DIR,
                split,
                "images",
                file
            )
        )

        shutil.copy2(
            mask_file,
            os.path.join(
                OUTPUT_DIR,
                split,
                "masks",
                file
            )
        )

        count += 1

    print(
        f"{split}: {scene} → {count} pairs"
    )


# -------------------------
# Perform split
# -------------------------

for scene in train_scenes:
    copy_scene(scene, "train")

for scene in val_scenes:
    copy_scene(scene, "val")

for scene in test_scenes:
    copy_scene(scene, "test")

print("\n==============================")
print("DATASET SPLIT COMPLETE")
print("==============================")