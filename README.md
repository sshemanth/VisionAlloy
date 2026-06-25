# 🏭 Automated Surface Defect Inspection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-00C7B7?style=for-the-badge)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Industrial%20AI-orange?style=for-the-badge)

</div>

---

## 📌 Overview

This project is an end-to-end, deep learning-based industrial inspection system designed to automate metallic surface defect detection using state-of-the-art computer vision models. 

By replacing manual, error-prone visual inspection workflows with an optimized **YOLOv8 pipeline**, the system delivers millisecond-level inference alongside actionable manufacturing analytics. The entire engine is wrapped in a high-fidelity, cinematic Streamlit industrial dashboard built for production operators and quality assurance managers alike.

> **Dataset Note:** Trained and validated using the industry-standard **NEU Surface Defect Dataset**.

---

## 🚀 Key Features

* **Real-Time Bounding Box Localization:** High-accuracy defect tracking powered by YOLOv8.
* **Automated Quality Decisions:** Instant sorting logic marking materials as `PASS`, `REVIEW`, or `REJECT`.
* **Dynamic Quality Scoring:** Algorithmic calculation of surface integrity based on defect density and severity.
* **Batch & Video Workflows:** Support for high-throughput image batches and continuous video streams.
* **Explainable AI (XAI):** Built-in heatmaps highlighting regions of interest for deep validation.
* **Industrial PDF Reporting:** Instant, automated generation of quality assurance reports via ReportLab.

---

## 🏭 Supported Defect Classes

The system is rigorously trained to segment and identify six core industrial metallic defects:

| Defect Type | Code | Description | Severity Threshold |
| :--- | :--- | :--- | :--- |
| **Crazing** | `Cr` | Fine, network-like surface cracking from thermal stress. | Medium |
| **Inclusion** | `In` | Foreign non-metallic matter embedded into the metal matrix. | High |
| **Patches** | `Pa` | Irregularly textured surface regions affecting coating adhesion. | Low |
| **Pitted Surface** | `Ps` | Small localized cavities or holes resulting from corrosion or wear. | High |
| **Rolled-in Scale** | `Rs` | Heavy mill scale pressed into the metal during rolling. | Medium |
| **Scratches** | `Sc` | Linear mechanical abrasions undermining structural finish. | Low / Medium |
