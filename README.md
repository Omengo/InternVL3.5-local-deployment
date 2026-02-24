InternVL3.5-2B Video Action Recognition for Battery Disassembly
This repository provides a comprehensive implementation and deployment guide for InternVL3.5-2B, specifically optimized for high-precision action recognition in Electric Vehicle (EV) battery disassembly tasks. The system leverages Chain-of-Thought (CoT) reasoning to identify complex industrial actions, such as bolt unscrewing, with high accuracy.

📋 Table of Contents
Environment

Detailed Tutorial

Core Scripts & Parameter Tuning

Inference & Testing

Project Insights

Contributors

Environment
Example Environment

Hardware

CPU: Intel Core i9-13900k or equivalent.

GPU: NVIDIA RTX 4090 24 GiB (Driver version >= 527.41).

Memory: 32GiB RAM (Minimum 16GB required for inference).

Software

OS: Windows 10/11 .

Python: 3.8.10 (Recommended for package compatibility).

CUDA: 12.x (PyTorch compatible).

PyTorch: 2.2.2.

Detailed Tutorial
1. Verifying GPU Configuration
Before installation, verify your CUDA installation and GPU status:

Bash
# Check CUDA version
nvcc -V

# Verify GPU and Quantization Library (bitsandbytes)
python scripts/debug_gpu.py
The debug_gpu.py script checks for CUDA availability and ensures the bitsandbytes library is correctly imported for 4-bit/8-bit quantization.

🛠 Installation Guide
Note: To maintain environment consistency, it is recommended to install these dependencies in your existing system environment. No new virtual environment is required.

1. Install PyTorch
Install the version compatible with CUDA 12.x:

Bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
2. Install Frameworks & Video Utilities
Bash
# Multi-modal libraries
pip install transformers accelerate timm

# Video processing (Essential for frame sampling)
pip install decord opencv-python pillow

📥 Model Download & Weight Setup
The model weights are the core of this system. Please follow these steps carefully:

1. Download Model Weights
Download the full InternVL3.5-2B weights from HuggingFace:

Link: OpenGVLab/InternVL3_5-2B

Instructions: Ensure you download all files, including .safetensors, config.json, tokenizer.json, and all .py script files.

2. Local Weight Placement
To ensure the scripts can find the model using relative paths, place the weights in the following directory:

Plaintext
C:\Users\LENOVO\InternVL\model\VL3.5-2b\

Bash
# Install PyTorch for CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install InternVL & Multi-modal Libraries
pip install transformers accelerate timm

# Install Video Processing Tools
pip install decord opencv-python pillow
Core Scripts & Parameter Tuning
A. High-Precision Battery Analysis (analysis_battery.py)
This script uses Chain-of-Thought (CoT) prompts to force the model to analyze worker hands, tools, and components before reaching a final conclusion.

How to adjust parameters:

CLIP_DURATION = 6.0: Set the window size in seconds. 6s is chosen to ensure a complete disassembly action is visible.

STRIDE = 2.0: The overlap interval. A 2s stride ensures high density so no momentary actions are missed.

NUM_FRAMES = 12: Number of frames sampled per clip. Increase this for videos with fast tool movements.

Run Command:

Bash
python scripts/analysis_battery.py
B. Real-Time Visualization (video_analysis_system.py)
Generates an annotated video with bounding boxes for actions, components, and tools, including live detection statistics.

How to adjust parameters:

DETECTION_INTERVAL = 5: Sets how often (in frames) the model performs detection. Set to 1 for every frame (slower) or higher for faster processing.

Run Command:

Bash
python scripts/video_analysis_system.py
C. Temporal Segmentation (temporal_analysis.py)
Generates a timestamped text report using a predefined vocabulary to ensure consistency.

How to adjust parameters:

SEGMENT_DURATION = 5: The duration of each report segment. Adjust based on the pace of the operation.

Run Command:

Bash
python scripts/temporal_analysis.py
Inference & Testing
Action Labeling Convention
All inference must strictly follow the project's standardized naming convention to ensure data alignment:

✅ Recommended: 0_unscrewing-bolt

❌ Legacy (Do Not Use): remove the cover

Testing Model Loading
To verify the model can load into VRAM and perform basic chat:

Bash
python scripts/test_minimal.py
Project Insights
Video Performance: InternVL3.5 shows stable results on video segments ranging from 5 seconds to 4.5 minutes.

Known Limitations: Inference on videos around the 3-minute mark has shown occasional instability; further optimization of sampling frames is ongoing.

Contributors
Qingfeng: Lead for InternVL3.5 research, local deployment, and CoT precision tuning.


