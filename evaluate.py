
import json
import os
import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from dataset import BrainTumourDataset
from unet_model import UNet


# 1. Settings
IMAGE_SIZE = 256
BATCH_SIZE = 2

DATASET_PATH = "data/DATASET-2/Segmentation"
MODEL_PATH = "models/best_unet_model.pth"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Evaluation device:", device)


# 2. Load test split
with open("splits.json", "r") as file:
    splits = json.load(file)

dataset = BrainTumourDataset(
    DATASET_PATH,
    image_size=IMAGE_SIZE
)

path_to_index = {
    (image, mask): index
    for index, (image, mask)
    in enumerate(dataset.image_paths)
}

test_items = splits["test"]

test_indices = [
    path_to_index[(item["image"], item["mask"])]
    for item in test_items
]

test_dataset = Subset(dataset, test_indices)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))


# 3. Load trained model
model = UNet().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# 4. Metric calculation
results = {
    "Glioma": [],
    "Meningioma": [],
    "Pituitary tumor": []
}

all_results = []


def calculate_metrics(pred, target):

    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()

    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)

    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)

    precision = (tp + 1e-7) / (tp + fp + 1e-7)

    recall = (tp + 1e-7) / (tp + fn + 1e-7)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall)
    }


# 5. Evaluate test images
with torch.no_grad():

    for batch_index, (images, masks) in enumerate(test_loader):

        images = images.to(device)

        outputs = model(images)

        predictions = (
            torch.sigmoid(outputs) > 0.5
        ).cpu().numpy()

        targets = masks.numpy()

        start = batch_index * BATCH_SIZE

        for j in range(len(images)):

            item = test_items[start + j]

            category = os.path.basename(
                os.path.dirname(item["image"])
            )

            metrics = calculate_metrics(
                predictions[j, 0],
                targets[j, 0]
            )

            results[category].append(metrics)
            all_results.append(metrics)

        if (batch_index + 1) % 25 == 0:
            print(
                f"Evaluated {(batch_index + 1) * BATCH_SIZE}"
                f"/{len(test_dataset)} images"
            )


# 6. Display results
def average_metric(data, key):
    return np.mean([item[key] for item in data])


print("\nFINAL TEST RESULTS")
print("=" * 45)

for category, data in results.items():

    print(f"\n{category}")
    print("Images:", len(data))

    for metric in ["dice", "iou", "precision", "recall"]:
        print(
            metric.capitalize() + ":",
            round(average_metric(data, metric), 4)
        )


print("\nOVERALL TEST RESULTS")
print("=" * 45)

for metric in ["dice", "iou", "precision", "recall"]:

    print(
        metric.capitalize() + ":",
        round(average_metric(all_results, metric), 4)
    )


# 7. Save results
with open("test_results.json", "w") as file:
    json.dump(
        {
            "overall": {
                metric: average_metric(all_results, metric)
                for metric in ["dice", "iou", "precision", "recall"]
            },
            "categories": {
                category: {
                    metric: average_metric(data, metric)
                    for metric in ["dice", "iou", "precision", "recall"]
                }
                for category, data in results.items()
            }
        },
        file,
        indent=4
    )

print("\nResults saved to test_results.json")
