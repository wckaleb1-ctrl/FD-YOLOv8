# FD-YOLOv8
# 🌊 FD-YOLOv8: Enhanced YOLOv8 for Floating Object Detection on Water Surfaces

FD-YOLOv8 is an improved YOLOv8-based detection framework designed for **robust and precise detection of floating objects on water surfaces**.  
The model enhances YOLOv8n through three key structural modules — **SPDConv**, **CSP_DCNv2CoordConv**, and **C2S_Upsample** — to address challenges such as small-object loss, reflection interference, and multi-scale feature degradation.

🔍 Model Overview

To address the challenges of small-object detection, visual similarity, and complex water-surface disturbances, our model introduces three key structural improvements based on YOLOv8n — forming the FD-YOLOv8 architecture:

SPDConv Module (Sparse and Dynamic Convolution)
Enhances contextual feature extraction through multi-branch sparse convolution and adaptive fusion.
By dynamically weighting different feature branches, SPDConv improves the model’s ability to discriminate between visually similar debris categories (e.g., glass vs. plastic bottles) and retains fine-grained details during downsampling.

CSP_DCNv2CoordConv Module
Integrates Deformable Convolution v2 (DCNv2) and Coordinate Convolution (CoordConv) to strengthen spatial perception and adaptability under dynamic lighting and wave disturbances.
This combination allows the model to adjust to irregular object shapes and illumination variations, significantly improving robustness against glare, ripples, and deformation on complex water surfaces.

C2S_Upsample Module (Channel-to-Spatial Upsample)
Achieves cross-scale feature fusion and super-resolution reconstruction during upsampling.
By reorganizing multi-level features and preserving high-frequency information, it enhances detection performance on small and distant floating objects, maintaining clear boundaries and high confidence even under multi-angle perspectives.

Together, these three modules enable FD-YOLOv8 to achieve superior precision, recall, and mAP performance compared to standard YOLOv8n, with improved robustness in small-object detection, reflection suppression, and class discrimination across diverse water environments.
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
<img width="243" height="172" alt="image" src="https://github.com/user-attachments/assets/c31fe7a4-17e0-43d8-8c33-bde4dc278302" />


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
