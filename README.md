# FD-YOLOv8
# 🌊 FD-YOLOv8: Enhanced YOLOv8 for Floating Object Detection on Water Surfaces

FD-YOLOv8 is an improved YOLOv8-based detection framework designed for **robust and precise detection of floating objects on water surfaces**.  
The model enhances YOLOv8n through three key structural modules — **SPDConv**, **CSP_DCNv2CoordConv**, and **C2S_Upsample** — to address challenges such as small-object loss, reflection interference, and multi-scale feature degradation.

---

## 🔍 Overview

Floating debris such as plastic, glass, and metal severely threaten aquatic ecosystems and urban water management.  
FD-YOLOv8 improves detection accuracy and robustness in dynamic water environments by introducing:

- **SPDConv**: Enhances contextual feature extraction and discrimination between visually similar debris (e.g., plastic vs. glass).  
- **CSP_DCNv2CoordConv**: Strengthens spatial localization and adapts to illumination, glare, and deformation.  
- **C2S_Upsample**: Refines upsampling for better small-object perception and high-frequency detail preservation.

---

## 🧩 Model Architecture

The overall architecture of FD-YOLOv8 is illustrated below:

![FD-YOLOv8 Architecture](<img width="691" height="536" alt="image" src="https://github.com/user-attachments/assets/35121819-52ad-428a-bde3-0c688ab2abad" />
)

---

## 📂 Dataset

The dataset used in this study includes **12,000 labeled images** of eight major categories of floating objects:

| Class | Description |
|:------|:-------------|
| Plastic | Bottles, containers, films |
| Glass | Transparent bottles, fragments |
| Metal | Cans, sheets |
| Wood | Branches, planks |
| Rubber | Tires, fragments |
| Foam | Styrofoam, packing materials |
| Net | Fishing nets, ropes |
| Bag | Plastic and woven bags |

**Composition:**
- Public **Roboflow floating-object dataset**  
- Self-collected images from **Nantong canal gates**  
- Synthetic samples generated using **Stable Diffusion 3**  
- Split ratio: **Train 70% / Val 20% / Test 10%**

---

## ⚙️ Environment Setup

```bash
# Recommended environment
Python 3.10
PyTorch 1.13.1
CUDA 11.6
NVIDIA Tesla T4 GPU

# Install dependencies
pip install -r requirements.txt

The repository contains some code and datasets of FD-YOLOv8
The relevant dataset code link is as follows. The file is shared via a network disk: data
Link: https://pan.baidu.com/s/1JmdZyNrXVrhWnjA7FDj2IQ?pwd=srx6 Extraction code: srx6
