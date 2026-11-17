
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


# 2. Load trained model
model = UNet().to(device)

checkpoint = torch.load(
    "models/best_unet_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Brain tumour segmentation model loaded!")


# 3. Get MRI image from user
image_path = input("\nEnter MRI image path: ").strip()

# Load MRI
original_image = Image.open(image_path).convert("L")

# Resize for model
image = original_image.resize(
    (256, 256),
    Image.Resampling.BILINEAR
)

image_array = np.array(
    image, dtype=np.float32
) / 255.0


# 4. Prepare input
input_tensor = torch.tensor(
    image_array,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0).to(device)


# 5. Predict tumour
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.sigmoid(output)

    predicted_mask = (
        prediction.squeeze().cpu().numpy() > 0.5
    )


# 6. Calculate tumour area
tumour_pixels = int(predicted_mask.sum())

total_pixels = predicted_mask.size

tumour_percentage = (
    tumour_pixels / total_pixels
) * 100


# 7. Print results
print("\n========== TUMOUR ANALYSIS ==========")

print("Total image pixels:", total_pixels)

print("Predicted tumour pixels:", tumour_pixels)

print(
    "Tumour area percentage:",
    round(tumour_percentage, 2), "%"
)

print("=====================================")


# 8. Create overlay
overlay = np.array(predicted_mask)

# 9. Display results
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].imshow(image_array, cmap="gray")
axes[0].set_title("Original MRI")

axes[1].imshow(predicted_mask, cmap="gray")
axes[1].set_title("Predicted Tumour Mask")

axes[2].imshow(image_array, cmap="gray")
axes[2].imshow(
    overlay,
    cmap="Reds",
    alpha=0.5
)
axes[2].set_title("Tumour Detection Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
