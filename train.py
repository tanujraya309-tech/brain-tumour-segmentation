
import os
import json
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from dataset import BrainTumourDataset
from unet_model import UNet


# --------------------------------
# 1. SETTINGS
# --------------------------------

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001

DATASET_PATH = "data/DATASET-2/Segmentation"
MODEL_FOLDER = "models"

os.makedirs(MODEL_FOLDER, exist_ok=True)


# --------------------------------
# 2. SELECT DEVICE
# --------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------------------
# 3. LOAD DATASET SPLITS
# --------------------------------

with open("splits.json", "r") as file:
    split_data = json.load(file)

full_dataset = BrainTumourDataset(
    DATASET_PATH,
    image_size=IMAGE_SIZE
)

# Match the saved split paths to dataset indices
path_to_index = {
    (image_path, mask_path): index
    for index, (image_path, mask_path)
    in enumerate(full_dataset.image_paths)
}


def create_subset(split_name):
    indices = []

    for item in split_data[split_name]:
        pair = (item["image"], item["mask"])
        indices.append(path_to_index[pair])

    return Subset(full_dataset, indices)


train_dataset = create_subset("train")
val_dataset = create_subset("validation")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# --------------------------------
# 4. INITIALIZE U-NET
# --------------------------------

model = UNet().to(device)


# --------------------------------
# 5. LOSS FUNCTIONS
# --------------------------------

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(logits, targets):

    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * targets).sum(
        dim=(1, 2, 3)
    )

    total = probabilities.sum(
        dim=(1, 2, 3)
    ) + targets.sum(
        dim=(1, 2, 3)
    )

    dice = (2 * intersection + 1) / (total + 1)

    return 1 - dice.mean()


def combined_loss(logits, targets):

    return (
        bce_loss(logits, targets)
        + dice_loss(logits, targets)
    )


# --------------------------------
# 6. OPTIMIZER
# --------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------
# 7. DICE METRIC
# --------------------------------

def calculate_dice(logits, targets):

    predictions = (torch.sigmoid(logits) > 0.5).float()

    intersection = (predictions * targets).sum(
        dim=(1, 2, 3)
    )

    total = predictions.sum(
        dim=(1, 2, 3)
    ) + targets.sum(
        dim=(1, 2, 3)
    )

    dice = (2 * intersection + 1) / (total + 1)

    return dice.mean().item()


# --------------------------------
# 8. TRAINING LOOP
# --------------------------------

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    train_loss_total = 0.0

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for batch_index, (images, masks) in enumerate(train_loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = combined_loss(outputs, masks)

        loss.backward()

        optimizer.step()

        train_loss_total += loss.item()

        if (batch_index + 1) % 50 == 0:
            print(
                f"Training batch {batch_index + 1}"
                f"/{len(train_loader)}"
            )

    average_train_loss = train_loss_total / len(train_loader)


    # --------------------------------
    # 9. VALIDATION
    # --------------------------------

    model.eval()

    val_loss_total = 0.0
    val_dice_total = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = combined_loss(outputs, masks)

            val_loss_total += loss.item()

            val_dice_total += calculate_dice(
                outputs, masks
            )

    average_val_loss = val_loss_total / len(val_loader)
    average_val_dice = val_dice_total / len(val_loader)


    # --------------------------------
    # 10. DISPLAY RESULTS
    # --------------------------------

    print(f"Training Loss: {average_train_loss:.4f}")
    print(f"Validation Loss: {average_val_loss:.4f}")
    print(f"Validation Dice: {average_val_dice:.4f}")


    # --------------------------------
    # 11. SAVE BEST MODEL
    # --------------------------------

    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_loss": average_val_loss,
                "validation_dice": average_val_dice
            },
            os.path.join(MODEL_FOLDER, "best_unet_model.pth")
        )

        print("Best model saved!")


print("\nTraining completed.")
