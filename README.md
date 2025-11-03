# NMT-EagleEyes

# 🤖 YOLOv8 Q-Block Detector & QC Pipeline

This repository contains the Jupyter Notebook (`eagle_eyes_train.ipynb`) used to train a custom YOLOv8 object detection model. The model is designed to detect "Q-blocks" on industrial images and then run a **Quality Control (QC) check** based on the detected count.

## ✨ Project Overview

The pipeline performs the following steps:

1.  **Dataset Download:** Downloads the latest version of the "eagle-eyes" dataset from Roboflow.
2.  **Model Training:** Trains a YOLOv8n model for 15 epochs on the custom dataset.
3.  **Validation:** Reports standard object detection metrics (mAP, Precision, Recall).
4.  **QC Testing:** Runs the best-trained model on a separate test set.
5.  **Business Logic:** Classifies each test image as **'GOOD'** or **'NOT\_GOOD'** by checking if the number of detected Q-blocks is exactly **14** (`EXPECTED_QBLOCK_COUNT`).
6.  **Accuracy Calculation:** Compares the QC classification (from detection count) against the image's filename ground truth (`_OK` or `_NG`) and reports the overall QC accuracy.

## 🚀 Setup and Requirements

### 1. Environment Setup

The notebook requires the `ultralytics` and `roboflow` libraries.

```bash
%pip install roboflow ultralytics
```

#### 2. Roboflow API Key

Ensure your Roboflow API key is active. The key is hardcoded in the notebook but should be moved to an environment variable (ROBOFLOW_API_KEY) for a production environment.

Parameter,Value,Description

data,'datasets/eagle-eyes-4/data.yaml',Path to the dataset configuration file.
epochs,15,The number of training epochs (was run for 1 in the current output).
model,'yolov8n.pt',The base model weights used for transfer learning.
EXPECTED_QBLOCK_COUNT,14,The core QC rule: An image is 'GOOD' if the model detects exactly 14 Q-blocks.