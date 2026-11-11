
import os
import numpy as np
from PIL import Image

folder = "data/DATASET-2/Segmentation/Glioma"

image = Image.open(os.path.join(folder, "enh_209.png"))
mask = Image.open(os.path.join(folder, "enh_209_mask.png"))

print("MRI dimensions:", image.size)
print("Mask dimensions:", mask.size)

mask_array = np.array(mask)

print("Mask pixel values:", np.unique(mask_array))
print("MRI image mode:", image.mode)
print("Mask image mode:", mask.mode)
