InternVL3.5 Video Action Recognition
This repository contains the implementation and deployment guide for InternVL3.5, focused on video-based action recognition for battery disassembly tasks. It specifically addresses identifying actions such as bolt unscrewing with high precision.

📋 Table of Contents
Hardware & Software Requirements

Installation

Deployment

Inference & Testing

Project Insights

Contributors

💻 Hardware & Software Requirements
Hardware
To ensure smooth inference and video processing, the following hardware is recommended:

GPU: NVIDIA RTX 4090 (24GB VRAM) or equivalent. The 2B model requires at least 8GB VRAM for bfloat16 inference.

System RAM: 32GB (Minimum 16GB).

Storage: At least 20GB of free disk space for model weights and dependencies.

OS: Windows 10/11 or Linux.

Python: 3.8.10 (Recommended for package compatibility).

CUDA: 12.x.

🛠 Installation
Note: To maintain environment consistency, it is recommended to use the existing system environment without creating a new virtual environment if preferred by the user.

1. Install PyTorch
Install the version compatible with CUDA 12.x:

2.1 Dependency Setup
This project is designed to run in your existing environment without the need for a new virtual environment.

Step 1: Install PyTorch (CUDA 12.x compatible)

Bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
Step 2: Install Multi-modal & Video Libraries

Bash
# Core Frameworks
pip install transformers accelerate timm

# Video Processing (Essential)
pip install decord opencv-python pillow
2.2 Environment Verification
Run the diagnostic script to ensure your GPU and libraries are correctly configured:

Bash
python scripts/debug_gpu.py
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

3. Model Preparation
Download Weights: Download the InternVL3.5-2B weights from HuggingFace.

Directory Structure: Create a model folder in the root directory.

Path: InternVL-Project/model/VL3.5-2b/.

Ensure all .json, .safetensors, and .py files from HuggingFace are inside this folder.

🚀 Core Features & Usage
A. High-Precision Battery Analysis (Recommended)
This mode uses a Chain-of-Thought (CoT) prompt to force the model to identify "Action-Component-Tool" triplets with scoring against ground truth.

Command: python scripts/analysis_battery.py.

Key Logic: Scans every 2s using a 6s window for maximum capture density.

B. Real-Time Detection & Visualization
Generates an annotated video with bounding boxes and real-time detection statistics.

Command: python scripts/video_analysis_system.py.

Output: Produces output_real_time_detection.mp4.

C. Temporal Segmentation Report
Generates a timestamped text report for easy documentation.

Command: python scripts/temporal_analysis.py.

Interval: Default is a 5s window for stable activity description.

📈 Project Insights
Video Performance: InternVL3.5 shows promising results on videos ranging from 5 seconds to 4.5 minutes.

Known Limitations: Inference results for videos around 3 minutes have shown some instability; further optimization is ongoing.

👥 Contributors
Qingfeng (Lead for InternVL3.5 research and deployment).


