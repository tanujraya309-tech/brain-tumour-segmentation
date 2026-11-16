
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import BrainTumourDataset
from unet_model import UNet


# 1. Select device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# 2. Load dataset
dataset = BrainTumourDataset(
    "data/DATASET-2/Segmentation",
    image_size=256
)

with open("splits.json", "r") as file:
    splits = json.load(file)


# 3. Get one validation image
path_to_index = {
    (image, mask): index
    for index, (image, mask) in enumerate(dataset.image_paths)
}

item = splits["validation"][0]

index = path_to_index[(item["image"], item["mask"])]

image, actual_mask = dataset[index]


# 4. Load trained U-Net
model = UNet().to(device)

checkpoint = torch.load(
    "models/best_unet_model.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# 5. Predict tumour mask
with torch.no_grad():

    input_image = image.unsqueeze(0).to(device)

    output = model(input_image)

    probabilities = torch.sigmoid(output)

    predicted_mask = (
        probabilities > 0.5
    ).float()


# 6. Convert tensors to NumPy
original = image.squeeze().numpy()

actual = actual_mask.squeeze().numpy()

predicted = predicted_mask.squeeze().cpu().numpy()


# 7. Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original, cmap="gray")
plt.title("Original MRI")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(actual, cmap="gray")
plt.title("Actual Tumour Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(predicted, cmap="gray")
plt.title("Predicted Tumour Mask")
plt.axis("off")

plt.tight_layout()
plt.show()


# 8. Calculate Dice for this image
intersection = np.sum(predicted * actual)

total = np.sum(predicted) + np.sum(actual)

dice = (2 * intersection + 1) / (total + 1)

print("Individual Image Dice Score:", round(float(dice), 4))

print("Actual tumour pixels:", int(np.sum(actual)))
print("Predicted tumour pixels:", int(np.sum(predicted)))

# 9. Calculate predicted tumour area

total_pixels = predicted.size

tumour_pixels = int(np.sum(predicted))

tumour_percentage = (
    tumour_pixels / total_pixels
) * 100

print("\nTUMOUR AREA ANALYSIS")
print("----------------------------")

print("Total MRI pixels:", total_pixels)

print("Predicted tumour pixels:", tumour_pixels)

print(
    "Tumour area percentage:",
    round(tumour_percentage, 2),
    "%"
)
