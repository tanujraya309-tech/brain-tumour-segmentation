
import os

categories = ["Glioma", "Meningioma", "Pituitary tumor"]

base_path = "data/DATASET-2/Segmentation"

for category in categories:
    folder = os.path.join(base_path, category)

    masks = [
        f for f in os.listdir(folder)
        if f.endswith("_mask.png")
    ]

    missing = []

    for mask in masks:
        original = mask.replace("_mask.png", ".png")

        if not os.path.exists(os.path.join(folder, original)):
            missing.append(mask)

    print("\nCategory:", category)
    print("Total masks:", len(masks))
    print("Missing MRI images:", len(missing))

