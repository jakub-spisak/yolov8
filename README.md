# Real-Time Object Detection with YOLOv8

This repository contains the implementation and evaluation of a real-time object detection system using **YOLOv8**, developed as part of a deep learning project by Jakub Spišák, Ľubomír Švec, and Daniel Zemančík.

## 📌 Overview

This project leverages **YOLOv8s**—a small, fast, and efficient variant of the YOLOv8 model—for object detection on the **COCO dataset**. The system is trained locally using an NVIDIA RTX 4060 GPU and evaluated across multiple computational backends. Key performance insights, architectural breakdowns, error analysis, and optimization tips are included.

## 🔧 Features

- End-to-end pipeline: data preprocessing, model training, evaluation
- Real-time performance with YOLOv8s
- Evaluation across multiple deployment formats: PyTorch, TorchScript, ONNX, MNN, NCNN
- Benchmark and error analysis included
- Based on the COCO dataset with class-wise performance insights

## 🚀 Model Training

- **Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Dataset**: [COCO 2017](https://cocodataset.org/#download)
- **Model**: `YOLOv8s` pretrained weights
- **Hardware**: RTX 4060 GPU, Ryzen 7 7840HS, 16GB RAM
- **Training duration**: ~108 hours

### Training Parameters

epochs: 100
batch: 16
imgsz: 640
seed: 69
resume: True
amp: False
dropout: 0.1
plots: True
optimizer: SGD (auto)
lr_schedule: Cosine
## 🧬 Detailed Model Topology

### 🏗 Backbone

- Based on improved **CSPDarknet** with **C2f** modules.
- Processes image features at multiple scales (P3, P4, P5).
- Uses **SiLU** activation.
- Integrates **SPPF (Spatial Pyramid Pooling - Fast)** for global context with minimal overhead.

### 🪢 Neck

- Combines **FPN** (top-down) and **PAN** (bottom-up) networks.
- Merges features from different scales using **C2f refinement**.
- Produces enhanced multi-resolution feature maps (P3′′, P4′′, P5′′).

### 🎯 Head

- **Anchor-free**, **decoupled design**:
  - **Classification branch**: class scores per cell
  - **Regression branch**: bounding box offsets
- Uses **Distribution Focal Loss (DFL)** with Reg-Max = 16
- Final predictions are filtered with **Non-Maximum Suppression (NMS)**

## 🧪 Experimental Results

| Metric        | best.pt (epoch 83) | last.pt (epoch 100) |
|---------------|--------------------|----------------------|
| mAP50–95      | 0.455              | 0.422                |
| mAP50         | 0.625              | 0.590                |
| Max F1        | 0.61               | 0.59                 |
| Optimal conf. | 0.287              | 0.323                |

### ⚙️ Deployment Benchmark (best.pt)

| Format      | Size (MB) | mAP50–95 | Latency (ms) | FPS    |
|-------------|-----------|-----------|---------------|--------|
| PyTorch     | 21.5      | 0.4548    | 7.3           | 137.03 |
| TorchScript | 43.0      | 0.4514    | 5.81          | 172.16 |
| ONNX        | 42.8      | 0.4514    | 8.52          | 117.36 |
| MNN         | 42.7      | 0.4513    | 60.92         | 16.42  |
| NCNN        | 42.7      | 0.4514    | 67.23         | 14.88  |

### 🔍 Key Findings

- **Small objects** like chairs and cups are harder to detect, especially in indoor settings.
- **Duplicate detections** of persons increase false positives.
- **Contextual confusion** due to mosaic augmentation misleads the model (e.g., zebras vs. cows).

For more detail information check the pdf document
