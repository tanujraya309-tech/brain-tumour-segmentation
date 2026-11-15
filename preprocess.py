import numpy as np
from PIL import Image

# Load MRI image and its tumour mask
mri = Image.open(
    "data/DATASET-2/Segmentation/Pituitary tumor/enh_59_mask.png"
)

mask = Image.open(
    "data/DATASET-2/Segmentation/Pituitary tumor/enh_59.png"
)

# Resize MRI and mask to 256 × 256
mri = mri.resize((256, 256), Image.Resampling.BILINEAR)
mask = mask.resize((256, 256), Image.Resampling.NEAREST)

# Convert images into NumPy arrays
mri_array = np.array(mri, dtype=np.float32) / 255.0
mask_array = np.array(mask)

# Convert tumour mask into binary format
binary_mask = (mask_array == 255).astype(np.float32)

# Display preprocessing information
print("MRI shape:", mri_array.shape)
print("Mask shape:", binary_mask.shape)
print("MRI pixel range:", mri_array.min(), "to", mri_array.max())
print("Mask values:", np.unique(binary_mask))

print("Preprocessing completed successfully!")