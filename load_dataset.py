
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Load MRI image
mri = Image.open("data/DATASET-2/Segmentation/Glioma/enh_209.png")

# 2. Load tumour mask
mask = Image.open("data/DATASET-2/Segmentation/Glioma/enh_209_mask.png")

# 3. Convert mask into NumPy array
mask_array = np.array(mask)
# Identify tumour pixels
tumour_region = mask_array == 255

# Calculate tumour area
tumour_pixels = np.sum(tumour_region)

# Calculate tumour percentage
tumour_percentage = (tumour_pixels / mask_array.size) * 100

print("Unique mask values:", np.unique(mask_array))
print("Tumour pixels:", tumour_pixels)
print("Tumour percentage:", tumour_percentage, "%")

# 4. Calculate tumour pixels
tumour_region = mask_array == 255
tumour_pixels = np.sum(tumour_region)

print("Tumour area:", tumour_pixels, "pixels")

# 5. Display original MRI and mask
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(mri, cmap="gray")
plt.title("Original Brain MRI")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask_array, cmap="gray")
plt.title("Tumour Mask")
plt.axis("off")

plt.show()

# 6. Display tumour overlay
plt.figure(figsize=(7, 7))

plt.imshow(mri, cmap="gray")
plt.imshow(tumour_region, cmap="Reds", alpha=0.5)

plt.title("Brain MRI with Tumour Overlay")
plt.axis("off")

plt.show()


# Dice Similarity Coefficient

ground_truth = tumour_region
prediction = tumour_region

intersection = np.sum(ground_truth & prediction)

dice_score = (2 * intersection) / (
    np.sum(ground_truth) + np.sum(prediction)
)

print("Dice Similarity Coefficient:", dice_score)
