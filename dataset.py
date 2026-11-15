
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class BrainTumourDataset(Dataset):

    def __init__(self, root_dir, image_size=256):

        self.root_dir = root_dir
        self.image_size = image_size
        self.image_paths = []

        categories = ["Glioma", "Meningioma", "Pituitary tumor"]

        for category in categories:

            folder = os.path.join(root_dir, category)

            for filename in os.listdir(folder):

                if filename.endswith("_mask.png"):
                    continue

                if filename.endswith(".png"):

                    image_path = os.path.join(folder, filename)

                    mask_name = filename.replace(".png", "_mask.png")
                    mask_path = os.path.join(folder, mask_name)

                    if os.path.exists(mask_path):
                        self.image_paths.append(
                            (image_path, mask_path)
                        )

        print("Total MRI-mask pairs found:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path, mask_path = self.image_paths[index]

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path)

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        image = np.array(image, dtype=np.float32) / 255.0

        mask = np.array(mask)
        mask = (mask == 255).astype(np.float32)

        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask
