# VisionAlloy — Automated Surface Defect Inspection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Industrial%20Dashboard-red?style=for-the-badge&logo=streamlit)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-00C7B7?style=for-the-badge)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Industrial%20AI-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)

### AI-Powered Industrial Surface Defect Detection Platform

Real-time metallic surface inspection using YOLOv8 object detection, intelligent quality assessment, industrial analytics, and automated inspection reporting.

</div>

---

# 📌 Overview

VisionAlloy is a deep learning-based industrial inspection system developed for automated metallic surface defect detection using computer vision and YOLOv8 object detection models.

The system replaces traditional manual inspection workflows with an AI-powered inspection dashboard capable of:

- Real-time defect localization
- PASS / REVIEW / REJECT decisions
- Surface quality scoring
- Industrial defect analytics
- Batch inspection workflows
- Video inspection support
- Explainable AI heatmaps
- Predictive maintenance insights
- Automated report generation

The project was developed using the **NEU Surface Defect Dataset** and deployed through a cinematic Streamlit industrial dashboard interface.

---

# 🏭 Supported Defect Classes

| Defect Type | Description |
|---|---|
| Crazing | Fine crack-like surface defects |
| Inclusion | Embedded foreign material defects |
| Patches | Irregular textured surface regions |
| Pitted Surface | Small holes and surface damage |
| Rolled-in Scale | Rolling process surface marks |
| Scratches | Linear abrasion-based defects |

---

# 🚀 Key Features

## 🔎 AI-Based Defect Detection
- YOLOv8s real-time object detection
- Bounding box localization
- Confidence score analysis
- Multiple defect detection support

---

## 🏭 Industrial Inspection Dashboard
- Cinematic industrial UI
- Real-time inference metrics
- PASS / REVIEW / REJECT decisions
- Severity classification system

---

## 📊 Factory Analytics
- Total inspections
- Reject rate analysis
- Defect trend visualization
- Average inference time monitoring
- Most common defect tracking

---

## 📦 Batch Inspection
- Multiple image processing
- Bulk quality inspection
- Batch analytics generation
- CSV report export

---

## 🎥 Video Inspection
- MP4 video processing
- Frame-by-frame defect detection
- Real-time inspection workflow

---

## 🧠 Explainable AI Heatmaps
- Visual attention mapping
- Defect focus region visualization
- Model interpretability support

---

## 🔥 Defect Heatmap Analytics
- Surface defect distribution analysis
- Frequent defect region visualization
- Industrial quality trend analysis

---

## 🛠 Smart Recommendation Engine

| Defect | Recommendation |
|---|---|
| Scratch | Surface polishing recommended |
| Inclusion | Inspect raw material purity |
| Pitted Surface | Check corrosion protection process |
| Rolled-in Scale | Inspect rolling equipment alignment |

---

## ⚠ Predictive Maintenance Insights

The system analyses recurring defects and predicts possible manufacturing issues:

- Roller misalignment
- Surface wear patterns
- Material contamination risks
- Process instability indicators

---

## 📄 PDF & CSV Report Export
- Inspection report generation
- Defect summary export
- Industrial QA documentation

---

## 🗂 Inspection History Logging
- Inspection ID tracking
- Historical inspection records
- Searchable inspection history
- Analytics persistence

---

# 🧱 System Architecture

```text
Input Image / Video
        ↓
YOLOv8s Detection Engine
        ↓
Defect Localization
        ↓
Confidence Analysis
        ↓
Severity Classification
        ↓
PASS / REVIEW / REJECT Decision
        ↓
Industrial Analytics Dashboard
        ↓
Report Generation & Logging
```

---

# 🧠 Deep Learning Model

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| YOLOv8n | 0.548 | 0.635 | 0.663 | 0.345 |
| YOLOv8s | 0.770 | 0.866 | 0.876 | 0.559 |
| YOLOv8m | 0.722 | 0.727 | 0.787 | 0.455 |

### Final Selected Model: YOLOv8s

YOLOv8s achieved the best balance between:
- localization accuracy
- computational efficiency
- real-time inference performance

---

# 📂 Project Structure

```text
VisionAlloy/
│
├── app.py
├── best.pt
├── requirements.txt
├── runtime.txt
├── README.md
│
├── assets/
├── reports/
├── outputs/
└── sample_images/
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/VisionAlloy.git
cd VisionAlloy
```

---

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 3. Run Application

```bash
streamlit run app.py
```

---

# 📦 Requirements

```text
streamlit
ultralytics
opencv-python-headless
torch
torchvision
numpy
pandas
pillow
matplotlib
reportlab
```

---

# 🌐 Streamlit Cloud Deployment

Create `runtime.txt`

```text
python-3.11
```

Then deploy directly from GitHub using Streamlit Cloud.

---

# 📸 Dashboard Modules

## 🏭 Inspection Dashboard
Single-image defect inspection and quality evaluation.

## 📦 Batch Inspection
Large-scale industrial quality inspection workflows.

## 📊 Factory Analytics
Production monitoring and industrial statistics.

## 🎥 Video Inspection
Continuous frame-based surface inspection.

## 🧠 Explainability Dashboard
AI heatmap visualization and model interpretation.

## 🕒 Inspection History
Inspection tracking and analytics logging.

---

# 📈 Dataset Information

### Dataset
NEU Surface Defect Dataset

### Classes
6 defect categories

### Total Images
1,800 images

### Image Resolution
200 × 200

### Training Split
- 70% Training
- 15% Validation
- 15% Testing

---

# 🔬 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| YOLOv8 | Object detection |
| Streamlit | Dashboard interface |
| OpenCV | Image processing |
| PyTorch | Deep learning |
| Pandas | Data analysis |
| NumPy | Numerical operations |

---

# ⚠️ Limitations

- Sensitive to extreme lighting variations
- Reflective metallic textures may cause false positives
- Limited dataset diversity
- Real-world industrial environments may contain unseen defect patterns

---

# 🔮 Future Improvements

- Edge AI deployment
- Live industrial camera integration
- Advanced Grad-CAM explainability
- Cloud database synchronization
- Industrial IoT integration
- Real-time conveyor belt monitoring

---

# 👨‍💻 Team VisionAlloy

| Team Member | Contribution |
|---|---|
| Sri Sai Hemanth Bollepalli | GUI, deployment, analytics, system integration |
| Harshitha Kolgatta Swamy | YOLO experimentation and model evaluation |
| Meet Jobanputra | Data engineering and backend integration |

---

# 📜 License

This project was developed for academic and research purposes.

---

<div align="center">

### VisionAlloy  
### Intelligent Industrial Surface Inspection Platform

Built with Deep Learning • Computer Vision • Industrial AI

</div>
