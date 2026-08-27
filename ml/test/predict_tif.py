import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ============================================================
# SETTINGS
# ============================================================

IMAGE_DIR = r"images"
MODEL_PATH = r"best_unet.pth"
OUTPUT_DIR = r"predictions"

TILE_SIZE = 256
THRESHOLD = 0.5

DEVICE = torch.device("cpu")


# ============================================================
# DOUBLE CONV
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.conv(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(1, 8)
        self.enc2 = DoubleConv(8, 16)
        self.enc3 = DoubleConv(16, 32)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(32, 64)

        self.up3 = nn.ConvTranspose2d(
            64, 32, 2, 2
        )

        self.dec3 = DoubleConv(
            64, 32
        )

        self.up2 = nn.ConvTranspose2d(
            32, 16, 2, 2
        )

        self.dec2 = DoubleConv(
            32, 16
        )

        self.up1 = nn.ConvTranspose2d(
            16, 8, 2, 2
        )

        self.dec1 = DoubleConv(
            16, 8
        )

        self.output = nn.Conv2d(
            8, 1, 1
        )


    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        b = self.bottleneck(
            self.pool(e3)
        )

        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("FULL SENTINEL-1 OIL SPILL PREDICTION")
print("=" * 60)

print("\nLoading model...")

model = UNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIND TIFF FILES
# ============================================================

tif_files = sorted([

    f for f in os.listdir(IMAGE_DIR)

    if f.lower().endswith(
        (".tif", ".tiff")
    )

])


print(
    f"\nFound {len(tif_files)} Sentinel-1 images."
)


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

for image_number, filename in enumerate(
    tif_files,
    start=1
):

    print("\n" + "=" * 60)

    print(
        f"IMAGE {image_number}/{len(tif_files)}"
    )

    print(
        f"Processing: {filename}"
    )

    print("=" * 60)


    # ========================================================
    # PATHS
    # ========================================================

    input_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    scene_name = os.path.splitext(
        filename
    )[0]

    scene_output_dir = os.path.join(
        OUTPUT_DIR,
        scene_name
    )

    os.makedirs(
        scene_output_dir,
        exist_ok=True
    )

    output_mask = os.path.join(
        scene_output_dir,
        "prediction_mask.tif"
    )

    output_overlay = os.path.join(
        scene_output_dir,
        "prediction_overlay.png"
    )


    # ========================================================
    # READ IMAGE
    # ========================================================

    with rasterio.open(
        input_path
    ) as src:

        image = src.read(1)

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

        height = src.height

        width = src.width


    print(
        f"Size: {width} × {height}"
    )

    print(
        f"CRS: {crs}"
    )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    valid = np.isfinite(image)

    minimum = np.percentile(
        image[valid],
        2
    )

    maximum = np.percentile(
        image[valid],
        98
    )

    image = np.clip(
        image,
        minimum,
        maximum
    )

    image = (
        image - minimum
    ) / (
        maximum - minimum + 1e-8
    )


    # ========================================================
    # OUTPUT ARRAYS
    # ========================================================

    prediction_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    prediction_count = np.zeros(
        (height, width),
        dtype=np.float32
    )


    # ========================================================
    # TILE PROCESSING
    # ========================================================

    rows = (
        height + TILE_SIZE - 1
    ) // TILE_SIZE

    columns = (
        width + TILE_SIZE - 1
    ) // TILE_SIZE

    total_tiles = rows * columns

    tile_number = 0


    print(
        f"Processing {total_tiles} tiles..."
    )


    for y in range(
        0,
        height,
        TILE_SIZE
    ):

        for x in range(
            0,
            width,
            TILE_SIZE
        ):

            tile_number += 1

            print(
                f"\rTile {tile_number}/{total_tiles}",
                end=""
            )


            y_end = min(
                y + TILE_SIZE,
                height
            )

            x_end = min(
                x + TILE_SIZE,
                width
            )


            tile = image[
                y:y_end,
                x:x_end
            ]


            tile_height = tile.shape[0]
            tile_width = tile.shape[1]


            # ------------------------------------------------
            # PAD TILE
            # ------------------------------------------------

            padded = np.zeros(
                (
                    TILE_SIZE,
                    TILE_SIZE
                ),
                dtype=np.float32
            )

            padded[
                :tile_height,
                :tile_width
            ] = tile


            # ------------------------------------------------
            # TENSOR
            # ------------------------------------------------

            tensor = torch.tensor(
                padded,
                dtype=torch.float32
            )

            tensor = tensor.unsqueeze(0)
            tensor = tensor.unsqueeze(0)


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            with torch.no_grad():

                output = model(
                    tensor
                )

                probability = torch.sigmoid(
                    output
                )


            probability = (
                probability
                .squeeze()
                .cpu()
                .numpy()
            )


            # ------------------------------------------------
            # REMOVE PADDING
            # ------------------------------------------------

            probability = probability[
                :tile_height,
                :tile_width
            ]


            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            prediction_sum[
                y:y_end,
                x:x_end
            ] += probability

            prediction_count[
                y:y_end,
                x:x_end
            ] += 1


    print(
        "\nTile processing complete."
    )


    # ========================================================
    # FINAL PROBABILITY MAP
    # ========================================================

    probability_map = (

        prediction_sum
        /
        np.maximum(
            prediction_count,
            1
        )

    )


    # ========================================================
    # BINARY MASK
    # ========================================================

    binary_mask = (

        probability_map >= THRESHOLD

    ).astype(
        np.uint8
    )


    # ========================================================
    # OIL STATISTICS
    # ========================================================

    oil_pixels = np.sum(
        binary_mask > 0
    )

    total_pixels = (
        width * height
    )

    oil_percentage = (

        oil_pixels
        /
        total_pixels
        *
        100

    )


    # Probability only for predicted oil pixels

    oil_probabilities = probability_map[
        binary_mask > 0
    ]


    if len(oil_probabilities) > 0:

        mean_oil_probability = float(
            np.mean(
                oil_probabilities
            )
        )

        max_oil_probability = float(
            np.max(
                oil_probabilities
            )
        )

    else:

        mean_oil_probability = 0.0

        max_oil_probability = 0.0


    # ========================================================
    # AREA
    # ========================================================

    pixel_width = abs(
        transform.a
    )

    pixel_height = abs(
        transform.e
    )

    pixel_area = (

        pixel_width
        *
        pixel_height

    )

    oil_area_m2 = (

        oil_pixels
        *
        pixel_area

    )

    oil_area_km2 = (

        oil_area_m2
        /
        1_000_000

    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("OIL SPILL RESULTS")
    print("-" * 40)

    print(
        f"Oil pixels: "
        f"{oil_pixels:,}"
    )

    print(
        f"Oil percentage: "
        f"{oil_percentage:.2f}%"
    )

    print(
        f"Spill area: "
        f"{oil_area_km2:.4f} km²"
    )

    print(
        f"Mean oil probability: "
        f"{mean_oil_probability:.4f}"
    )

    print(
        f"Maximum oil probability: "
        f"{max_oil_probability:.4f}"
    )

    print(
        f"Threshold: "
        f"{THRESHOLD}"
    )


    # ========================================================
    # SAVE MASK
    # ========================================================

    profile.update(

        dtype=rasterio.uint8,

        count=1,

        compress="lzw"

    )


    with rasterio.open(

        output_mask,

        "w",

        **profile

    ) as dst:

        dst.write(
            binary_mask,
            1
        )


    # ========================================================
    # SAVE OVERLAY
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    ax.imshow(
        image,
        cmap="gray"
    )

    overlay = np.ma.masked_where(
        binary_mask == 0,
        binary_mask
    )

    ax.imshow(
        overlay,
        cmap="Reds",
        alpha=0.45
    )

    ax.set_title(
        f"Oil Spill Prediction - {scene_name}"
    )

    ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_overlay,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"Saved: {output_mask}"
    )

    print(
        f"Saved: {output_overlay}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)

print(
    "ALL IMAGES PROCESSED"
)

print("=" * 60)

print(
    f"Results saved in: "
    f"{os.path.abspath(OUTPUT_DIR)}"
)

print("\nDONE.")