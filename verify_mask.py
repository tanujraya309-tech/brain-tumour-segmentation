
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load one MRI image and its mask
mri = Image.open(
    "data/DATASET-2/Segmentation/Pituitary tumor/enh_59_mask.png"
)

mask = Image.open(
    "data/DATASET-2/Segmentation/Pituitary tumor/enh_59.png"
)

# Convert into NumPy arrays
mri_array = np.array(mri)
mask_array = np.array(mask)

# Identify tumour pixels
tumour = mask_array == 255

# Display MRI, mask and overlay
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(mri_array, cmap="gray")
plt.title("Original MRI")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask_array, cmap="gray")
plt.title("Original Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mri_array, cmap="gray")
plt.imshow(tumour, cmap="Reds", alpha=0.5)
plt.title("Tumour Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Mask values:", np.unique(mask_array))
print("Tumour pixels:", np.sum(tumour))
