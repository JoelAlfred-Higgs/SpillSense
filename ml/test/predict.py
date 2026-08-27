import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ============================================================
# SETTINGS
# ============================================================

IMAGE_DIR = r"dataset_split/test/images"
MASK_DIR = r"dataset_split/test/masks"

MODEL_PATH = r"best_unet.pth"

DEVICE = torch.device("cpu")

THRESHOLD = 0.5

NUMBER_OF_IMAGES = 5


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

print("Loading model...")

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
# FIND TEST TILES
# ============================================================

files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".npy")
])


print(
    f"\nTest tiles available: {len(files)}"
)


# ============================================================
# SELECT TILES CONTAINING OIL
# ============================================================

selected = []

for filename in files:

    mask = np.load(
        os.path.join(
            MASK_DIR,
            filename
        )
    )

    if np.sum(mask > 0) > 0:

        selected.append(filename)

    if len(selected) >= NUMBER_OF_IMAGES:

        break


print("\nSelected tiles:")

for filename in selected:

    print(filename)


# ============================================================
# PREDICT
# ============================================================

for filename in selected:

    print(
        f"\nProcessing: {filename}"
    )

    image = np.load(
        os.path.join(
            IMAGE_DIR,
            filename
        )
    )

    mask = np.load(
        os.path.join(
            MASK_DIR,
            filename
        )
    )

    image_tensor = torch.tensor(
        image,
        dtype=torch.float32
    )

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():

        output = model(
            image_tensor
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

    prediction = (
        probability >= THRESHOLD
    ).astype(np.uint8)


    # ========================================================
    # STATISTICS
    # ========================================================

    print(
        f"Ground truth oil pixels: "
        f"{np.sum(mask > 0)}"
    )

    print(
        f"Predicted oil pixels: "
        f"{np.sum(prediction > 0)}"
    )

    print(
        f"Maximum probability: "
        f"{probability.max():.4f}"
    )

    print(
        f"Mean probability: "
        f"{probability.mean():.4f}"
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    axes[0].imshow(
        image,
        cmap="gray"
    )

    axes[0].set_title(
        "Sentinel-1 VV"
    )

    axes[1].imshow(
        mask,
        cmap="gray"
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[2].imshow(
        probability,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "Oil Probability"
    )

    axes[3].imshow(
        prediction,
        cmap="gray"
    )

    axes[3].set_title(
        f"Prediction > {THRESHOLD}"
    )

    for ax in axes:

        ax.axis("off")

    plt.suptitle(
        filename
    )

    plt.tight_layout()

    plt.show()