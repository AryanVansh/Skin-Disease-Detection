# 🧴 Skin Disease Detection

An AI-powered system for classifying skin lesions into seven disease categories using deep learning. This project leverages the **HAM10000** dataset, a fine-tuned **ResNet50** model, **Albumentations** for preprocessing, and a **Gradio** interface for real-time predictions.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Directory Structure](#-directory-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Model Training](#2-model-training)
  - [3. Run the Web App](#3-run-the-web-app)
- [File Descriptions](#-file-descriptions)
- [Sample Prediction](#-sample-prediction)
- [Dependencies](#-dependencies)
- [License](#-license)
- [Contributing](#-contributing)
- [Acknowledgements](#-acknowledgements)

---

## 📖 Overview

Skin cancer is among the most common forms of cancer globally. Early detection and classification of skin lesions are critical for effective treatment and better outcomes. This project trains a convolutional neural network (CNN) to classify dermatoscopic images into seven disease types.

> ✅ Features:
> - Data cleaning & preprocessing pipeline
> - Custom training on ResNet50
> - Gradio-powered interface for predictions
> - Modular code for flexibility and reuse

---

## 📊 Dataset

We use the [**HAM10000 Dataset**](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), which contains 10,015 dermatoscopic images of skin lesions.

### Disease Categories:

| Label | Description                        |
|-------|------------------------------------|
| `nv`  | Melanocytic nevi                   |
| `mel` | Melanoma                           |
| `bkl` | Benign keratosis-like lesions      |
| `bcc` | Basal cell carcinoma               |
| `akiec`| Actinic keratoses / Bowen’s disease |
| `vasc`| Vascular lesions                   |
| `df`  | Dermatofibroma                     |

---

## 🧠 Model Architecture

- **Backbone**: ResNet50 (pre-trained on ImageNet)
- **Custom Head**:
  - Global Average Pooling
  - Dropout for regularization
  - Dense layers for classification

The model is compiled with:
- **Loss**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Metric**: Accuracy

---

## 📂 Directory Structure

├── app.py # Gradio web interface ├── predict.py # Prediction pipeline using the trained model ├── model.py # ResNet50 model with custom classification head ├── dataset.py # Dataset creation with augmentation ├── preprocessing.py # Albumentations transforms ├── train.py # Model training logic ├── reclean.py # Clean CSVs and remove corrupted images ├── unzip.py # Unzip archive ├── check.py # Image file integrity checker ├── trial.py # Print CSV structure (debug) ├── best_model.keras # Pretrained model file ├── Skin_data.txt # Dataset source link


---

## 💻 Installation

### Step 1: Clone the repository

```
git clone https://github.com/yourusername/skin-disease-detection.git
cd skin-disease-detection
```

### Step 2: Create a virtual environment (optional but recommended)
```
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

Step 3: Install dependencies
```
pip install -r requirements.txt
```
Or manually:
```
pip install tensorflow opencv-python pandas albumentations gradio
```
## 🚀 Usage
1. Data Preparation
  - Download the dataset from Kaggle: HAM10000

  - Extract using:
```
python unzip.py
```
  - Clean corrupted/missing image records:
```
python reclean.py
```
2. Model Training
```
python train.py
```
The model is trained on cleaned data and saved as best_model.keras under the models/ directory.

3. Run the Web App
```
python app.py
```
This will launch a Gradio interface in your browser. Upload an image and get the predicted skin disease type.

## 📄 File Descriptions


| File | Description                                                  |
|-------|-------------------------------------------------------------|
| `app.py`  | Launches Gradio web interface                           |
| `predict.py` | Loads trained model and predicts class               |
| `model.py` | Defines CNN architecture using ResNet50                |
| `dataset.py	` | 	Creates TensorFlow datasets with augmentation     |
| `preprocessing.py	`| Defines Albumentations preprocessing pipeline  |
| `train.py	`| Trains model using training and validation data        |
| `reclean.py	`| Removes invalid or unreadable image records from CSV |
| `unzip.py	`  | Unpacks the dataset archive                          |
| `check.py	` | Validates a single image's readability                |
| `Skin_data.txt	` | Source link to dataset on Kaggle                |


## 🖼️ Sample Prediction
Once the app is running, you’ll see something like:
```
AI-Based Skin Disease Detector
Upload a skin lesion image and get the predicted disease.
```
Just drag and drop an image, and the predicted category (e.g., "Melanoma (mel)") will be shown.
## 📦 Dependencies
```
tensorflow
opencv-python
albumentations
pandas
gradio
```
You can install all dependencies using pip install -r requirements.txt

## 📄 requirements.txt
```
tensorflow>=2.8.0
opencv-python
albumentations
pandas
gradio
numpy
```
