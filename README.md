# VisionAlloy – Automated Surface Defect Detection System

## Project Overview

VisionAlloy is an AI-powered automated surface defect detection system developed using deep learning and computer vision techniques. The project focuses on detecting and classifying steel surface defects using YOLOv8 object detection models trained on the NEU Surface Defect Dataset.

The system was designed to support industrial quality inspection processes by automatically identifying surface defects such as scratches, patches, crazing, inclusion, pitted surfaces, and rolled-in scale defects.

The project includes:

* Deep learning model development using YOLOv8
* Object detection training and evaluation
* AWS SageMaker-based model training
* Streamlit-based graphical user interface (GUI)
* GitHub deployment integration
* Industrial-style automated inspection workflow

---

# Objectives

The main objectives of the project are:

* Develop an automated defect detection system using deep learning
* Compare different YOLOv8 architectures
* Improve defect detection accuracy for industrial inspection
* Build an interactive GUI for real-time predictions
* Deploy the trained model using Streamlit Cloud

---

# Dataset Information

## Dataset Used

NEU Surface Defect Dataset

## Dataset Type

Object Detection Dataset

## Number of Classes

6 defect classes:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

## Dataset Split

The dataset was manually divided into:

* 70% Training
* 15% Validation
* 15% Testing

## Dataset Structure

```text
SplitDataset/
 ├── train/
 │    ├── images/
 │    └── labels/
 │
 ├── valid/
 │    ├── images/
 │    └── labels/
 │
 ├── test/
 │    ├── images/
 │    └── labels/
 │
 └── data.yaml
```

---

# Model Architectures

Three YOLOv8 architectures were evaluated:

| Model   | Description                                              |
| ------- | -------------------------------------------------------- |
| YOLOv8n | Nano version with lightweight architecture               |
| YOLOv8s | Small version with balanced performance                  |
| YOLOv8m | Medium version with larger feature extraction capability |

---

# Experimental Settings

| Parameter               | Setting            |
| ----------------------- | ------------------ |
| Task Type               | Object Detection   |
| Image Size              | 640 × 640          |
| Epochs                  | 200                |
| Optimizer               | AdamW              |
| Early Stopping          | Enabled            |
| Early Stopping Patience | 25                 |
| Confidence Threshold    | 0.25               |
| Framework               | Ultralytics YOLOv8 |
| Training Platform       | AWS SageMaker      |
| GUI Framework           | Streamlit          |

---

# Model Performance

| CNN Architecture | Train Accuracy | Validation Accuracy | Test Accuracy |
| ---------------- | -------------: | ------------------: | ------------: |
| YOLOv8n          |         67.90% |              67.68% |        66.26% |
| YOLOv8s          |         87.40% |              84.58% |        87.56% |
| YOLOv8m          |         78.85% |              77.84% |        78.71% |

## Best Model

YOLOv8s achieved the highest overall performance and provided the best balance between detection accuracy and computational efficiency.

---

# Technologies Used

## Programming Language

* Python

## Deep Learning Frameworks

* PyTorch
* Ultralytics YOLOv8

## Libraries

* OpenCV
* NumPy
* Pandas
* Pillow
* Matplotlib
* Streamlit

## Development Environment

* AWS SageMaker
* Jupyter Notebook
* GitHub

---

# Project Structure

```text
VisionAlloy/
 ├── app.py
 ├── requirements.txt
 ├── runtime.txt
 ├── best.pt
 ├── README.md
 ├── SplitDataset/
 │    ├── train/
 │    ├── valid/
 │    └── test/
 │
 ├── runs/
 │    └── detect/
 │
 └── notebooks/
      └── Main.ipynb
```

---

# GUI Features

The project includes a Streamlit-based GUI for real-time surface defect inspection.

## GUI Capabilities

* Upload steel surface images
* Perform real-time defect detection
* Adjustable confidence threshold
* Defect classification table
* PASS/REJECT inspection decision
* Visual bounding box predictions

---

# Installation Guide

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/VisionAlloy.git
```

## Navigate to Project Folder

```bash
cd VisionAlloy
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

## Launch Streamlit GUI

```bash
streamlit run app.py
```

The application will open in:

```text
http://localhost:8501
```

---

# Model Prediction Workflow

1. User uploads a steel surface image
2. Streamlit GUI receives image input
3. YOLOv8 model performs object detection
4. Defects are identified and classified
5. Bounding boxes are generated
6. PASS/REJECT decision is displayed
7. Detection results are shown in the GUI

---

# Example Defects Detected

The system can detect:

* Surface scratches
* Inclusion defects
* Pitted surfaces
* Rolled-in scale defects
* Surface patches
* Crazing cracks

---

# Deployment

The project was deployed using:

* GitHub Repository
* Streamlit Community Cloud

The deployment uses the trained YOLOv8s model generated in AWS SageMaker.

---

# Challenges Faced

Several challenges were encountered during development:

* AWS SageMaker proxy limitations
* Model deployment integration
* Streamlit cloud dependency compatibility
* Dataset restructuring for train/validation/test splitting
* GPU memory limitations for larger YOLO models

---

# Future Improvements

Potential future improvements include:

* Real-time webcam inspection
* Industrial camera integration
* Edge device deployment
* Faster inference optimization
* Multi-material surface inspection
* Advanced defect severity estimation

---

# Conclusion

VisionAlloy successfully demonstrates the effectiveness of deep learning-based object detection for industrial steel surface inspection. The YOLOv8-based system achieved strong detection performance while providing an easy-to-use graphical interface for real-time defect inspection.

The project highlights how AI and computer vision can improve industrial quality assurance processes by reducing manual inspection effort and increasing defect detection consistency.

---

# Authors

## VisionAlloy Team

* Sri Sai Hemanth Bollepalli
* Meet Jobanputra
* Harshitha Kolgatta Swamy

---

# License

This project was developed for academic and educational purposes.

---

# Acknowledgements

* Ultralytics YOLOv8
* AWS SageMaker
* Streamlit
* NEU Surface Defect Dataset
* Roboflow
