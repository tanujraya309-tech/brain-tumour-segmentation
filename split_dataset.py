
import os
import json
import random

from dataset import BrainTumourDataset

# Reproducible random split
random.seed(42)

# Load all MRI-mask pairs
dataset = BrainTumourDataset(
    "data/DATASET-2/Segmentation"
)

# Group image-mask pairs by tumour category
categories = {
    "Glioma": [],
    "Meningioma": [],
    "Pituitary tumor": []
}

for image_path, mask_path in dataset.image_paths:

    category = os.path.basename(
        os.path.dirname(image_path)
    )

    categories[category].append({
        "image": os.path.relpath(image_path),
        "mask": os.path.relpath(mask_path)
    })

# Create train, validation and test groups
splits = {
    "train": [],
    "validation": [],
    "test": []
}

print("\nDATASET SPLIT RESULTS")
print("-" * 45)

for category, pairs in categories.items():

    # Sort first, then shuffle reproducibly
    pairs.sort(key=lambda x: x["image"])
    random.shuffle(pairs)

    total = len(pairs)

    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    train = pairs[:train_end]
    validation = pairs[train_end:val_end]
    test = pairs[val_end:]

    splits["train"].extend(train)
    splits["validation"].extend(validation)
    splits["test"].extend(test)

    print(f"\n{category}")
    print("Training:", len(train))
    print("Validation:", len(validation))
    print("Testing:", len(test))

# Save the split information
with open("splits.json", "w") as file:
    json.dump(splits, file, indent=4)

print("\n" + "-" * 45)
print("TOTAL DATASET")
print("Training:", len(splits["train"]))
print("Validation:", len(splits["validation"]))
print("Testing:", len(splits["test"]))
print("\nSplit information saved to splits.json")
