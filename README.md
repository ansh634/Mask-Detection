# Mask Detection Project

This project is a machine learning-based application that detects whether a person is wearing a mask or not. It has potential applications in areas such as hospitals, public places, and workplaces to ensure safety and compliance with health guidelines.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Screenshots](#screenshots)
9. [Future Work](#future-work)

---

## Overview
The mask detection project is designed to determine whether a person is wearing a mask by analyzing input images or real-time camera feeds. It uses TensorFlow and OpenCV to process images and classify them as "Mask" or "No Mask."

---

## Features
- Real-time mask detection using a webcam.
- High accuracy with trained machine learning models.
- Easy-to-use interface for image or video feed analysis.
- Visualization of results with bounding boxes around detected faces.

---

## Technologies Used
- **Python**
- **TensorFlow** for training and model deployment.
- **OpenCV** for image processing and video stream handling.
- **Scikit-learn** for data preprocessing and evaluation.
- **Matplotlib** and **Seaborn** for data visualization.
- **NumPy** for numerical operations.

---

## Dataset
The dataset contains labeled images of individuals with masks and without masks. Data collection and preprocessing were crucial steps in ensuring the model's performance. Images were resized, augmented, and normalized before feeding them into the model.

---

## Model Architecture
- **Base Model**: MobileNetV2 (pretrained on ImageNet).
- **Custom Layers**: Fully connected layers for binary classification (Mask/No Mask).
- **Loss Function**: Binary Cross-Entropy.
- **Optimizer**: Adam Optimizer.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/mask-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd mask-detection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the dataset or ensure the dataset is placed in the `data/` directory.

---

## Usage
1. Train the model:
   ```bash
   python train.py
   ```
2. Test the model with static images:
   ```bash
   python detect_image.py --image path/to/image.jpg
   ```
3. Real-time detection using a webcam:
   ```bash
   python detect_webcam.py
   ```

---

## Screenshots

### Real-Time Mask Detection
![Real-Time Detection](/Images/home.png)
![Real-Time Detection](/Images/withoutMask.png)
![Real-Time Detection](/Images/with_Mask.png)



---

## Future Work
- Expand dataset to include diverse scenarios and demographics.
- Improve detection accuracy under challenging conditions like low light and occlusion.
- Develop a mobile application for mask detection.

---


Feel free to contribute to this project by raising issues or submitting pull requests!

