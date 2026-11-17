
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from unet_model import UNet


# 1. Select device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# 2. Load dataset split
with open("splits.json", "r") as f:
    splits = json.load(f)


# 3. Load trained U-Net model
model = UNet().to(device)

checkpoint = torch.load(
    "models/best_unet_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Trained model loaded successfully!")


# 4. Select tumour categories
categories = ["Glioma", "Meningioma", "Pituitary tumor"]


# 5. Visualize one image from each category
for category in categories:

    # Get validation images for this category
    category_samples = [
        item for item in splits["validation"]
        if category in item["image"]
    ]

    # Select second meningioma image, first for others
    if category == "Meningioma":
        sample = category_samples[1]
    else:
        sample = category_samples[0]

    print("\nCategory:", category)
    print("Image:", sample["image"])

    # Load MRI image
    image = Image.open(sample["image"]).convert("L")
    image = image.resize(
        (256, 256),
        Image.Resampling.BILINEAR
    )

    image_array = np.array(
        image, dtype=np.float32
    ) / 255.0

    # Load actual tumour mask
    mask = Image.open(sample["mask"])
    mask = mask.resize(
        (256, 256),
        Image.Resampling.NEAREST
    )

    actual_mask = np.array(mask) == 255

    # Prepare MRI for prediction
    input_tensor = torch.tensor(
        image_array,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)

    # Predict tumour mask
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output)

        predicted_mask = (
            prediction.squeeze().cpu().numpy() > 0.5
        )

    # Calculate prediction errors
    true_positive = predicted_mask & actual_mask
    false_positive = predicted_mask & ~actual_mask
    false_negative = ~predicted_mask & actual_mask

    # Create error overlay
    error_overlay = np.zeros((256, 256, 3))

    error_overlay[true_positive] = [0, 1, 0]   # Green
    error_overlay[false_positive] = [1, 0, 0]  # Red
    error_overlay[false_negative] = [0, 0, 1]  # Blue

    # Calculate Dice score
    intersection = np.logical_and(
        predicted_mask, actual_mask
    ).sum()

    total_pixels = (
        predicted_mask.sum() + actual_mask.sum()
    )

    dice = (
        2 * intersection / total_pixels
        if total_pixels > 0 else 1.0
    )

    print("Dice Score:", round(dice, 4))
    print("Actual tumour pixels:", actual_mask.sum())
    print("Predicted tumour pixels:", predicted_mask.sum())

    # Display results
    fig, axes = plt.subplots(
        1, 4, figsize=(16, 4)
    )

    axes[0].imshow(image_array, cmap="gray")
    axes[0].set_title("Original MRI")

    axes[1].imshow(actual_mask, cmap="gray")
    axes[1].set_title("Actual Mask")

    axes[2].imshow(predicted_mask, cmap="gray")
    axes[2].set_title("Predicted Mask")

    axes[3].imshow(image_array, cmap="gray")
    axes[3].imshow(error_overlay, alpha=0.6)
    axes[3].set_title("Prediction Errors")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{category} | Dice: {dice:.4f}"
    )

    plt.tight_layout()
    plt.show()
