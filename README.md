# Brain Tumour Segmentation Using Deep Learning

### A 2D MRI-Based Tumour Segmentation and Area Estimation System Using U-Net

**Author:** Tanuj Raya  
**Field:** Artificial Intelligence and Computational Biology  
**Project Type:** Deep Learning | Medical Image Analysis | Computer Vision  
**Status:** Baseline implementation completed

---

## 1. Project Overview

This project implements a deep-learning-based system for brain tumour segmentation using 2D Magnetic Resonance Imaging (MRI) images.

A lightweight U-Net architecture is trained to identify tumour regions and generate pixel-level segmentation masks.

The system accepts an MRI image as input and produces:

- A predicted tumour segmentation mask.
- A visual overlay highlighting the predicted tumour region.
- The estimated tumour area in pixels.
- The percentage of the processed image occupied by the predicted tumour.

The primary objective is to explore how deep learning can assist with medical image segmentation and quantitative image analysis.

**Important:** This is an educational and portfolio research prototype. It is not clinically validated and must not be used for medical diagnosis or treatment decisions.

## 2. Project Objectives

1. Prepare and preprocess brain MRI images and corresponding segmentation masks.
2. Develop a custom U-Net architecture using PyTorch.
3. Train the model to distinguish tumour pixels from background pixels.
4. Evaluate segmentation performance using standard metrics.
5. Generate tumour masks from MRI images without requiring ground-truth masks during inference.
6. Estimate tumour area using the predicted segmentation mask.

## 3. Dataset

The project uses the segmentation portion of a brain tumour MRI dataset.

Three categories are included:

| Tumour category | Image-mask pairs |
|---|---:|
| Glioma | 554 |
| Meningioma | 708 |
| Pituitary tumour | 930 |
| **Total** | **2,192** |

The dataset contains 2,192 MRI images and 2,192 corresponding segmentation masks.

### Dataset split

The dataset was divided into training, validation and testing subsets using category-stratified image-level splitting with random seed 42.

| Dataset | Images |
|---|---:|
| Training | 1,533 |
| Validation | 328 |
| Testing | 331 |
| **Total** | **2,192** |

**Note:** Patient-level splitting could not be performed because patient identifiers were unavailable.

## 4. Data Preprocessing

The following preprocessing operations were performed:

- Conversion of MRI images to grayscale.
- Resizing images to 256 × 256 pixels.
- Normalisation of image intensities to the range [0, 1].
- Conversion of segmentation masks into binary masks.
- Nearest-neighbour interpolation for mask resizing.
- Verification of image-mask pairing.

The model uses a single grayscale MRI channel as input and generates a single-channel binary segmentation output.

## 5. Model Architecture

A custom **2D U-Net** architecture was implemented using PyTorch.

The architecture consists of:

- Encoder: 16, 32 and 64 feature channels.
- Bottleneck: 128 feature channels.
- Decoder: transposed convolutions and skip connections.
- Double convolution blocks using 3 × 3 convolutions, batch normalisation and ReLU activation.
- Final 1 × 1 convolution for binary segmentation.

### Why U-Net?

U-Net was selected because:

- It is designed for pixel-level image segmentation.
- Skip connections help preserve spatial information.
- Its encoder-decoder structure supports localisation of tumour regions.
- A lightweight implementation is suitable for experimentation on local hardware.

The current model performs binary segmentation only. It does not classify the tumour into glioma, meningioma or pituitary categories.

## 6. Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Model | Custom 2D U-Net |
| Input resolution | 256 × 256 |
| Batch size | 4 |
| Epochs | 10 |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Loss function | BCE + Dice Loss |
| Hardware | Apple M2 |
| Backend | Metal Performance Shaders (MPS) |

The best model checkpoint was selected using validation performance.

## 7. Model Performance

The trained model was evaluated on 331 held-out test images.

### Overall test results

| Evaluation metric | Score |
|---|---:|
| Dice coefficient | 0.5914 |
| Intersection over Union (IoU) | 0.4852 |
| Precision | 0.7618 |
| Recall | 0.5985 |

These values represent macro-averages of per-image metrics.

### Category-wise performance

| Category | Dice | IoU |
|---|---:|---:|
| Glioma | 0.4928 | 0.3912 |
| Meningioma | 0.4703 | 0.3637 |
| Pituitary tumour | 0.7431 | 0.6344 |

### Observations

- Pituitary tumour images achieved the strongest segmentation performance.
- Meningioma images showed the weakest Dice score.
- Precision was higher than recall overall, indicating that the model missed a meaningful proportion of tumour pixels.
- Further training, validation and model improvements are required.

These results represent an initial baseline, not clinical-level performance.

## 8. MRI Prediction and Tumour Area Estimation

After training, the model can accept a new 2D MRI image without its corresponding segmentation mask.

The prediction pipeline:

1. Loads the MRI image.
2. Converts it to grayscale.
3. Resizes it to 256 × 256.
4. Normalises the image.
5. Loads the trained U-Net checkpoint.
6. Generates a predicted tumour probability map.
7. Applies a threshold of 0.5 to obtain a binary mask.
8. Calculates the predicted tumour pixel count and image percentage.

### Area calculation

The predicted tumour area in pixels is:

\[
A = \sum_{x,y} \hat{M}(x,y)
\]

The percentage of the processed image occupied by the predicted tumour is:

\[
A_{\%} =
\frac{A}{256 \times 256} \times 100
\]

A sample validation prediction produced:

- Predicted tumour pixels: 1,306
- Processed image pixels: 65,536
- Estimated area percentage: 1.99%
- Individual Dice score: 0.8902

This is an individual example, not the overall test performance.

**Limitation:** The area is reported in pixels, not mm². Physical area estimation requires image pixel-spacing information. A single 2D slice also cannot establish tumour volume.

## 9. Project Structure

```text
brain-tumour-segmentation/
│
├── data/
│   └── DATASET-2/
│       └── Segmentation/
│           ├── Glioma/
│           ├── Meningioma/
│           └── Pituitary tumor/
│
├── models/
│   └── best_unet_model.pth
│
├── notebooks/
│   └── 01_mri_exploration.ipynb
│
├── dataset.py
├── split_dataset.py
├── splits.json
├── unet_model.py
├── train.py
├── predict.py
├── evaluate.py
├── visualize_results.py
├── analyze_mri.py
│
├── README.md
├── LICENSE
└── .gitignore
```

The dataset and model checkpoint may be excluded from the public repository because of storage limitations and dataset redistribution restrictions.

## 10. Running the Project

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd brain-tumour-segmentation
```

### Step 2: Install dependencies

```bash
pip install torch torchvision
pip install numpy matplotlib pillow scikit-learn
```

Install any additional dependencies required by the project scripts.

### Step 3: Prepare the dataset

Place the dataset inside the expected local directory:

```text
data/DATASET-2/Segmentation/
```

Ensure that each MRI image has its corresponding segmentation mask.

### Step 4: Train the model

```bash
python train.py
```

### Step 5: Evaluate the model

```bash
python evaluate.py
```

### Step 6: Visualise segmentation results

```bash
python visualize_results.py
```

### Step 7: Analyse a new MRI image

```bash
python analyze_mri.py
```

Follow the script's prompt for the MRI image path.

## 11. Limitations

The current implementation has several limitations:

- Moderate overall segmentation performance.
- Lower performance for glioma and meningioma categories.
- Image-level rather than patient-level dataset splitting.
- Limited training duration of ten epochs.
- No independent external dataset validation.
- Current inference pipeline supports 2D PNG/JPG images.
- Physical tumour area and 3D volume are not calculated.
- No clinical validation has been performed.

## 12. Future Improvements

Potential extensions include:

1. Patient-level dataset splitting.
2. Hyperparameter optimisation and extended training.
3. Data augmentation.
4. Improved loss functions for class imbalance.
5. Transfer learning using pretrained segmentation models.
6. Comparison with Attention U-Net, U-Net++ and SegFormer.
7. DICOM and NIfTI image support.
8. Physical area estimation using pixel-spacing metadata.
9. External dataset validation.
10. Uncertainty estimation for identifying unreliable predictions.

Tumour classification is intentionally outside the scope of this project and may be explored as a separate project.

## 13. Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Pillow
- Scikit-learn
- Visual Studio Code
- Git and GitHub

## 14. Conclusion

This project demonstrates the development of a complete baseline deep-learning pipeline for brain tumour segmentation using 2D MRI images.

A custom U-Net model was trained, evaluated and integrated into an image-only inference workflow capable of generating tumour masks and estimating tumour area in pixels.

The current results provide a foundation for further exploration of deep learning in medical image analysis, with particular emphasis on segmentation quality, generalisation and responsible interpretation.

---

**Author:** Tanuj Raya  
**Project:** Brain Tumour Segmentation Using Deep Learning  
**Field:** Artificial Intelligence and Computational Biology

*This project is intended for educational and research exploration only. It is not a clinically validated diagnostic system.*

