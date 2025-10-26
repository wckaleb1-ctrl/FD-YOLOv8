# 🌊 FD-YOLOv8: Enhanced YOLOv8 for Floating Object Detection on Water Surfaces

FD-YOLOv8 is an improved YOLOv8-based detection framework designed for **robust and precise detection of floating objects on water surfaces**.  
The model enhances YOLOv8n through three key structural modules — **SPDConv**, **CSP_DCNv2CoordConv**, and **C2S_Upsample** — to address challenges such as small-object loss, reflection interference, and multi-scale feature degradation.

---

## 🔍 Model Overview

To address the challenges of small-object detection, visual similarity, and complex water-surface disturbances, our model introduces three key structural improvements based on YOLOv8n — forming the **FD-YOLOv8 architecture**:

### 1️⃣ SPDConv Module (Sparse and Dynamic Convolution)
Enhances contextual feature extraction through multi-branch sparse convolution and adaptive fusion.  
By dynamically weighting different feature branches, SPDConv improves the model’s ability to discriminate between **visually similar debris categories** (e.g., glass vs. plastic bottles) and retains fine-grained details during downsampling.

### 2️⃣ CSP_DCNv2CoordConv Module
Integrates **Deformable Convolution v2 (DCNv2)** and **Coordinate Convolution (CoordConv)** to strengthen **spatial perception and adaptability** under dynamic lighting and wave disturbances.  
This combination allows the model to adjust to irregular object shapes and illumination variations, significantly improving robustness against **glare, ripples, and deformation** on complex water surfaces.

### 3️⃣ C2S_Upsample Module (Channel-to-Spatial Upsample)
Achieves **cross-scale feature fusion** and **super-resolution reconstruction** during upsampling.  
By reorganizing multi-level features and preserving high-frequency information, it enhances detection performance on **small and distant floating objects**, maintaining clear boundaries and high confidence even under multi-angle perspectives.

> Together, these three modules enable FD-YOLOv8 to achieve superior precision, recall, and mAP performance compared to standard YOLOv8n, with improved robustness in small-object detection, reflection suppression, and class discrimination across diverse water environments.

---

<div align="center">
  <img width="850" alt="FD-YOLOv8 Overview" src="https://github.com/user-attachments/assets/e2901410-a7d0-4216-b40e-f90123bbaa71" />
  <br>
  <em>Figure 1. Overview of FD-YOLOv8 framework and its three core structural improvements.</em>
</div>

---

## 🌍 Overview

Floating debris such as plastic, glass, and metal severely threaten aquatic ecosystems and urban water management.  
FD-YOLOv8 improves detection accuracy and robustness in dynamic water environments by introducing:

- **SPDConv**: Enhances contextual feature extraction and discrimination between visually similar debris (e.g., plastic vs. glass).  
- **CSP_DCNv2CoordConv**: Strengthens spatial localization and adapts to illumination, glare, and deformation.  
- **C2S_Upsample**: Refines upsampling for better small-object perception and high-frequency detail preservation.

---

## 🧩 Model Architecture

The overall architecture of **FD-YOLOv8** and its improved components are illustrated below.

<div align="center">
  <img width="850" alt="FD-YOLOv8 Architecture" src="https://github.com/user-attachments/assets/35121819-52ad-428a-bde3-0c688ab2abad" />
  <br>
  <em>Figure 2. The overall structure of FD-YOLOv8 model.</em>
</div>

<div align="center">
  <img width="500" alt="SPDConv Module" src="https://github.com/user-attachments/assets/eb763163-9618-4ec0-9206-19658da9862c" />
  <br>
  <em>Figure 3. SPDConv module structure — multi-branch sparse convolution and dynamic fusion.</em>
</div>

<div align="center">
  <img width="520" alt="CSP_DCNv2CoordConv Module" src="https://github.com/user-attachments/assets/50c4aa0e-2908-4877-ab86-5ef356d2aa11" />
  <br>
  <em>Figure 4. CSP_DCNv2CoordConv module — combining deformable convolution and coordinate convolution for enhanced spatial awareness.</em>
</div>

<div align="center">
  <img width="550" alt="C2S_Upsample Module" src="https://github.com/user-attachments/assets/89a04971-6e58-43c8-a4fc-8b8f2864f3c4" />
  <br>
  <em>Figure 5. C2S_Upsample module — channel-to-spatial reconstruction and multi-scale feature fusion.</em>
</div>

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

<div align="center">
  <img width="480" alt="Floating Object Dataset Samples" src="https://github.com/user-attachments/assets/c31fe7a4-17e0-43d8-8c33-bde4dc278302" />
  <br>
  <em>Figure 6. Sample images from the floating object dataset covering eight debris categories.</em>
</div>

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
