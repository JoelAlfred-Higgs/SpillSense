import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler


# ============================================================
# SETTINGS
# ============================================================

TRAIN_IMAGE_DIR = r"dataset_split/train/images"
TRAIN_MASK_DIR = r"dataset_split/train/masks"

VAL_IMAGE_DIR = r"dataset_split/val/images"
VAL_MASK_DIR = r"dataset_split/val/masks"

MODEL_PATH = r"best_unet.pth"

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-3

IMAGE_SIZE = 256

DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


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

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        mask_path = os.path.join(
            self.mask_dir,
            filename
        )

        image = np.load(image_path)
        mask = np.load(mask_path)

        # Image:
        # (256, 256) → (1, 256, 256)
        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        # Mask:
        # (256, 256) → (1, 256, 256)
        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask


# ============================================================
# LOAD DATA
# ============================================================

train_dataset = OilSpillDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR
)

val_dataset = OilSpillDataset(
    VAL_IMAGE_DIR,
    VAL_MASK_DIR
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# ============================================================
# BALANCED TRAINING SAMPLER
# ============================================================

sample_weights = []

for filename in train_dataset.files:

    mask_path = os.path.join(
        TRAIN_MASK_DIR,
        filename
    )

    mask = np.load(mask_path)

    # Check whether this tile contains oil
    if np.sum(mask > 0) > 0:
        weight = 5.0
    else:
        weight = 1.0

    sample_weights.append(weight)


sample_weights = torch.DoubleTensor(
    sample_weights
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    "Balanced training sampler enabled."
)


# ============================================================
# SMALL U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

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

        self.up3 = nn.ConvTranspose2d(
            64, 32, 2, 2
        )

        self.dec3 = DoubleConv(64, 32)

        self.up2 = nn.ConvTranspose2d(
            32, 16, 2, 2
        )

        self.dec2 = DoubleConv(32, 16)

        self.up1 = nn.ConvTranspose2d(
            16, 8, 2, 2
        )

        self.dec1 = DoubleConv(16, 8)

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
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.smooth = 1.0

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(
            predictions
        )

        predictions = predictions.view(-1)

        targets = targets.view(-1)

        intersection = (
            predictions * targets
        ).sum()

        dice = (
            2 * intersection
            + self.smooth
        ) / (
            predictions.sum()
            + targets.sum()
            + self.smooth
        )

        return 1 - dice


# ============================================================
# LOSS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()

dice_loss = DiceLoss()


def combined_loss(predictions, targets):

    bce = bce_loss(
        predictions,
        targets
    )

    dice = dice_loss(
        predictions,
        targets
    )

    return bce + dice


# ============================================================
# METRICS
# ============================================================

def dice_score(predictions, targets):

    predictions = torch.sigmoid(
        predictions
    )

    predictions = (
        predictions > 0.5
    ).float()

    intersection = (
        predictions * targets
    ).sum()

    dice = (
        2 * intersection + 1
    ) / (
        predictions.sum()
        + targets.sum()
        + 1
    )

    return dice.item()


def iou_score(predictions, targets):

    predictions = torch.sigmoid(
        predictions
    )

    predictions = (
        predictions > 0.5
    ).float()

    intersection = (
        predictions * targets
    ).sum()

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum()

    iou = (
        intersection + 1
    ) / (
        union + 1
    )

    return iou.item()


# ============================================================
# MODEL
# ============================================================

model = UNet().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

print("\nStarting training...\n")


for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------

    model.train()

    train_loss = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, masks in progress:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(images)

        loss = combined_loss(
            predictions,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        progress.set_postfix(
            loss=loss.item()
        )

    train_loss /= len(train_loader)


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    val_loss = 0
    val_dice = 0
    val_iou = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)

            loss = combined_loss(
                predictions,
                masks
            )

            val_loss += loss.item()

            val_dice += dice_score(
                predictions,
                masks
            )

            val_iou += iou_score(
                predictions,
                masks
            )

    val_loss /= len(val_loader)

    val_dice /= len(val_loader)

    val_iou /= len(val_loader)


    # -------------------------
    # PRINT RESULTS
    # -------------------------

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f}"
    )

    print(
        f"Val Dice:   {val_dice:.4f}"
    )

    print(
        f"Val IoU:    {val_iou:.4f}"
    )


    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "✓ Best model saved!"
        )


print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

print(
    f"Best model: {MODEL_PATH}"
)