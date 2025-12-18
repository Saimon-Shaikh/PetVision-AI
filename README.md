# PetVision-AI 🐾

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)

**A high-performance, offline desktop application for real-time Cat vs. Dog classification, powered by Deep Learning.**

---

## 📖 Description

**PetVision-AI** is an end-to-end Computer Vision project that demonstrates the complete lifecycle of a machine learning product. Moving beyond simple notebook experiments, this project takes a raw dataset and transforms it into a production-grade Windows application.

The core is a custom Convolutional Neural Network (CNN) trained on over 34,000 images, achieving **99% accuracy**. To ensure the application runs smoothly on standard laptops (without requiring a GPU), the model was optimized using **ONNX Runtime**, reducing memory usage and inference latency significantly.

The interface is built with a modern, dark-themed Tkinter GUI and packaged as a standalone executable using PyInstaller.

## 📸 Visuals

<img width="425" height="700" alt="Image" src="https://github.com/user-attachments/assets/680d7341-6881-41da-aa04-edfaf1f96e7d" /> <img width="425" height="700" alt="Image" src="https://github.com/user-attachments/assets/f50e1b0b-7a8f-4641-b041-4e174b9de3ab" />

*The modern, dark-mode interface allows users to upload images and get instant predictions with confidence scores.*

## ✨ Key Features

* **High Precision:** Achieves **>99% confidence** on validation datasets.
* **Lightning Fast:** Optimized with **ONNX Runtime** for millisecond inference speeds on CPU.
* **Modern UI:** A clean, "Dark Mode" graphical interface built with Python's Tkinter.
* **Offline Capable:** Runs locally with no internet connection required; data privacy is ensured.
* **Portable:** Deployed as a single `.exe` file (Windows) requiring no Python installation.

## 📂 Project Structure

```text
PetVision-AI/
├── models/
│   └── cat_dog_model.onnx      # Optimized inference model
├── app_onnx.py                 # Main styled GUI application
├── test_onnx.py                # Standalone script for model validation
├── train.py                    # PyTorch training script (source)
├── requirements.txt            # Project dependencies
├── app_icon.ico                # Application branding icon
├── screenshot.png              # Preview image for README
└── .gitignore                  # Prevents uploading build/temp files

## 🛠️ Step-by-Step Implementation

This project was built in five distinct engineering stages:

### 1. Data Engineering
Processed a large dataset of 34,000+ images (Cats & Dogs).
* **Cleaning:** Removed corrupted files and non-image headers.
* **Balancing:** Ensured equal distribution of classes to prevent bias.
* **Transformation:** Resized inputs to 128x128 and normalized RGB values.

### 2. Model Architecture
Designed a custom Convolutional Neural Network (CNN) using **PyTorch**.
* **Layers:** 3 Convolutional blocks with ReLU activation and Max Pooling.
* **Classifier:** Fully connected layers with Dropout to prevent overfitting.
* **Training:** Trained for 20 epochs using the Adam optimizer.



### 3. Optimization (ONNX)
Converted the heavy PyTorch (`.pth`) model into the **ONNX** (Open Neural Network Exchange) format.
* **Result:** Removed the dependency on the 1GB+ PyTorch library for the final app.
* **Performance:** Solved precision mismatches by enforcing specific Float32 operations.

### 4. GUI Development
Built a user-friendly frontend using `tkinter`.
* **Design:** Custom styling for a professional "Software" look (Segoe UI fonts, dark palette).
* **Integration:** Connected the GUI directly to the ONNX inference engine.

### 5. Deployment
Packaged the entire environment using **PyInstaller**.
* Bundled dependencies and assets (icons, models) into a standalone executable.
* Configured relative paths to ensure stability across different Windows machines.

## 💻 Installation (For Developers)

If you want to run the source code or train the model yourself:

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Saimon-Shaikh/PetVision-AI.git](https://github.com/Saimon-Shaikh/PetVision-AI.git)
    cd PetVision-AI
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### Running from Source
To launch the optimized GUI application:
```bash
python app_onnx.py
