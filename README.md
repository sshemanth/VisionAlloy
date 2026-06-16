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

# 🔮 Future Improvements

- Edge AI deployment
- Live industrial camera integration
- Advanced Grad-CAM explainability
- Cloud database synchronization
- Industrial IoT integration
- Real-time conveyor belt monitoring
