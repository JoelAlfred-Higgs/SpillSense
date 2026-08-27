import os
import rasterio

IMAGE_DIR = r"images"
MASK_DIR = r"masks"

image_files = {
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".tif")
}

mask_files = {
    f for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".tif")
}

print("IMAGE / MASK CHECK")
print("=" * 50)

for filename in sorted(image_files):

    image_path = os.path.join(IMAGE_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename)

    print(f"\n{filename}")

    if filename not in mask_files:
        print("  ❌ MASK NOT FOUND")
        continue

    with rasterio.open(image_path) as img:
        print(
            f"  Image: {img.width} × {img.height} | "
            f"CRS: {img.crs} | Bands: {img.count}"
        )

    with rasterio.open(mask_path) as mask:
        print(
            f"  Mask:  {mask.width} × {mask.height} | "
            f"CRS: {mask.crs} | Bands: {mask.count}"
        )

print("\n" + "=" * 50)
print("CHECK COMPLETE")