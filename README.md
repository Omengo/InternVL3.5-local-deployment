# InternVL3.5-local-deployment
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
CPU: Intel Core i9-13900k or equivalent.

GPU: NVIDIA RTX 4090 24GiB (Driver version >= 527.41).

Memory: 64GiB.

Software
OS: Windows 10/Linux.

Python: 3.8.10 (Recommended for package compatibility).

CUDA: 12.x.

🛠 Installation
Note: To maintain environment consistency, it is recommended to use the existing system environment without creating a new virtual environment if preferred by the user.

1. Install PyTorch
Install the version compatible with CUDA 12.x:

Bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
2. Install OpenMMLab Tools
Bash
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0
mim install mmdet
3. Install MMAction2 from Source
Bash
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
🚀 Deployment
Configuration
Update your configuration file (e.g., swinlarge.py) to include visualization backends:

Python
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend')
]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)
🧪 Inference & Testing
Running Inference
To run action recognition on a specific video (e.g., testclip_6.mp4), use the following command:

Bash
python VisualCM.py \
    ./configs/recognition/swin/swinlarge.py \
    ./checkpoints/best_acc_epoch.pth \
    ./data/testclip_6.mp4 \
    --out-video ./outputs/result.mp4 \
    --label-file ./datasets/label_file.txt
Action Labeling Convention
Ensure all labels follow the project naming convention:

Use 0_unscrewing-bolt instead of "remove the cover".

📈 Project Insights
Video Performance: InternVL3.5 shows promising results on videos ranging from 5 seconds to 4.5 minutes.

Known Limitations: Inference results for videos around 3 minutes have shown some instability; further optimization is ongoing.

👥 Contributors
Qingfeng (Lead for InternVL3.5 research and deployment).

Matthew (Collaborator for VideoLLaMA2 and Repo Maintenance).
