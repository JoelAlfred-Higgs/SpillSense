import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


IMAGE_DIR = r"dataset_split/test/images"
MASK_DIR = r"dataset_split/test/masks"
MODEL_PATH = r"best_unet.pth"

BATCH_SIZE = 16
DEVICE = torch.device("cpu")


# ============================================================
# DATASET
# ============================================================

class OilSpillDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".npy")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        image = np.load(
            os.path.join(
                self.image_dir,
                filename
            )
        )

        mask = np.load(
            os.path.join(
                self.mask_dir,
                filename
            )
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(1, 8)
        self.enc2 = DoubleConv(8, 16)
        self.enc3 = DoubleConv(16, 32)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(32, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec3 = DoubleConv(64, 32)

        self.up2 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.dec2 = DoubleConv(32, 16)

        self.up1 = nn.ConvTranspose2d(16, 8, 2, 2)
        self.dec1 = DoubleConv(16, 8)

        self.output = nn.Conv2d(8, 1, 1)

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
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# LOAD
# ============================================================

dataset = OilSpillDataset(
    IMAGE_DIR,
    MASK_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = UNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()


# ============================================================
# EVALUATE
# ============================================================

intersection = 0
union = 0
pred_pixels = 0
true_pixels = 0

threshold = 0.5

with torch.no_grad():

    for images, masks in loader:

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= threshold
        ).float()

        intersection += (
            predictions * masks
        ).sum().item()

        union += (
            predictions
            + masks
            - predictions * masks
        ).sum().item()

        pred_pixels += predictions.sum().item()
        true_pixels += masks.sum().item()


# ============================================================
# RESULTS
# ============================================================

dice = (
    2 * intersection
    / (pred_pixels + true_pixels + 1e-8)
)

iou = (
    intersection
    / (union + 1e-8)
)

print("\n==============================")
print("TEST RESULTS")
print("==============================")

print(
    f"Test tiles: {len(dataset)}"
)

print(
    f"True oil pixels: {int(true_pixels)}"
)

print(
    f"Predicted oil pixels: {int(pred_pixels)}"
)

print(
    f"Dice: {dice:.4f}"
)

print(
    f"IoU:  {iou:.4f}"
)

print(
    f"Threshold: {threshold}"
)