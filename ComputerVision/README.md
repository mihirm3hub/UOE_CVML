# 🧠 UOE_CVML – Computer Vision Coursework

This repository contains practical coursework for the ECMM426 module, focused on hands-on applications of **Computer Vision (CV)** using Python. All notebooks are designed for clarity, experimentation, and step-by-step understanding.

---

## 📁 Repository Structure

```
UOE_CVML/
├── ComputerVision/
│   ├── ECMM426_CW_q1-q5.ipynb   ← CV tasks: preprocessing, filtering, CNN classification
│   └── ECMM426_CW_q6.ipynb      ← CV task: object detection using YOLOv8
├── README.md
└── requirements.txt             ← Add your dependencies here
```

---

## 🔍 Computer Vision Overview

The `ComputerVision/` folder features two main notebooks:

### `ECMM426_CW_q1-q5.ipynb`

- Covers foundational CV techniques:
    - Image loading, color space conversions (RGB, HSV, grayscale)
    - Gaussian blur, sharpening, and morphological filters
    - Canny edge detection, Sobel filters
    - Thresholding with Otsu's method
- Applies a **PyTorch CNN** for classifying image patches
- Visual analysis of predictions and misclassifications

### `ECMM426_CW_q6.ipynb`

- Implements an object detection pipeline using **YOLOv8** (Ultralytics)
    - Handles zip extraction and image annotation
    - Detects objects via pretrained YOLO model
    - Evaluates detection quality using bounding box visualization and class labels

---

## 📊 Models & Evaluation

### ✅ Models Used

- **Convolutional Neural Network (CNN)**:  
  Built using PyTorch for classification of image segments.

- **YOLOv8 (Ultralytics)**:  
  State-of-the-art real-time object detection model with pre-trained weights.

---

### 📈 Evaluation Metrics

- **Accuracy**: Measures classification correctness
- **Confusion Matrix**: Summarizes prediction vs. ground truth
- **Precision & Recall**: Key for object detection assessment
- **IoU (Intersection-over-Union)**: Used to measure bounding box overlap for detection

---

### 💬 Prediction Comments

- The PyTorch CNN achieved **solid accuracy**, with performance improving after applying preprocessing steps like normalization and augmentation.
- YOLOv8 worked **effectively out-of-the-box** with minimal fine-tuning, handling shape detection and classification with high reliability.
- Notable challenges:
    - Differentiating similar shapes under poor lighting
    - Noise handling and filter tuning
    - Bounding box misalignment in edge cases

---

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mihirm3hub/UOE_CVML.git
   cd UOE_CVML/ComputerVision
   ```

2. **Install required packages**:
   *(create `requirements.txt` if not present)*
   ```bash
   pip install opencv-python numpy matplotlib torch torchvision ultralytics
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

---

## 📷 Example Output (Suggested Additions)

Consider adding sample figures like:
- Original vs. filtered image
- Edge detection results
- YOLOv8 detection bounding boxes
- CNN confusion matrix heatmap

---

## ✨ Contributions

This project is intended for academic and prototyping purposes. You’re welcome to fork, modify, and extend it—especially if exploring more advanced models or novel datasets.

---

## 📬 Contact

Created by [@mihirm3hub](https://github.com/mihirm3hub)  
For coursework under the ECMM426 module – University of Exeter
