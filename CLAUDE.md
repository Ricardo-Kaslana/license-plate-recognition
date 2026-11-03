# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a two-stage license plate recognition (LPR) system for Chinese license plates using:
1. **YOLOv8-Pose** - Detects license plates and outputs 4 corner keypoints for perspective correction
2. **CNN-Transformer** - Recognizes the license plate characters from corrected images

The system includes a Flask web application for user uploads (images/videos) and displays detection/annotation results.

## Architecture

### Two-Stage Pipeline
- **Stage 1 (Detection)**: YOLOv8-Pose detects license plates and predicts 4 corner keypoints (top-left, top-right, bottom-right, bottom-left). These corners are used for perspective transformation to correct skew/rotation.
- **Stage 2 (Recognition)**: The corrected plate image (48x160) is fed through a CNN visual encoder (ResNet-like blocks) to extract feature maps, then a Transformer decoder autoregressively generates character sequences.

### Key Components
- `model.py` - `LPR_Transformer` class: CNN encoder (4 BasicBlock layers, 16x downsampling) + Transformer decoder with positional encoding
- `dataset.py` - `CCPD_LPR_Dataset`: Parses CCPD2019 filenames containing bbox corners and label indices, applies perspective correction during loading
- `inference.py` - End-to-end inference pipeline combining YOLO detection + LPR recognition
- `lpr_app/` - Flask web application with user auth (Flask-Login), SQLite DB, and upload handling

### Data Flow
1. Raw image → YOLOv8-Pose → bbox + 4 corners
2. Corners → Perspective transform → 48x160 corrected plate
3. Corrected plate → CNN encoder → feature sequence (30 tokens for 3x10 feature map)
4. Transformer decoder → character sequence (autoregressive, stops at `</s>`)

### Character Set
Chinese provinces (31) + letters + digits. Special tokens: `<s>` (start), `</s>` (end), `<pad>` (padding). Max sequence length is 10 (8 chars + 2 tokens).

## Common Commands

### Environment Setup
```bash
# Install dependencies (Windows/PyTorch CUDA 12.1 wheels configured in pyproject.toml)
uv sync
# or
pip install -r requirements.txt
```

### Dataset Preparation
```bash
# Convert CCPD2019 to YOLO-Pose format (images + labels)
python prepare_yolo_pose_data.py
```

### Training

**YOLOv8-Pose Detection Model:**
```bash
# Using shell script
./train_yolo_pose.sh

# Or directly with yolo CLI
yolo train model=./yolov8n-pose.pt data=./yolo_ccpd_pose_dataset/ccpd_pose_data.yaml epochs=100 batch=16 imgsz=640 project=LPR_Pose_Detection name=yolov8n_ccpd_pose_e100
```

**LPR Transformer Recognition Model:**
```bash
python train_lpr.py
# Saves best model to checkpoints/best_lpr_model.pth
# Outputs training logs and visualizations to ./LPR_Rec/training_run_*/
```

### Inference
```bash
# Single image
python inference.py --input test_car_1.jpg --mode image

# Video processing
python inference.py --input demo.mp4 --mode video
```

### End-to-End Evaluation
```bash
python evaluate_e2e.py --test_dir your_test_dir
```

### Web Application
```bash
python run.py
# Runs Flask dev server on default port
# Uploads stored in uploads/, processed results in processed/
```

## Important Notes

### Font Setup
On Ubuntu, install Chinese fonts for matplotlib visualization:
```bash
apt-get install fonts-wqy-zenhei
```

### Model Weight Locations
- YOLOv8-Pose: `checkpoints/best_yolov8_pose_model_latest.pt`
- LPR Transformer: `checkpoints/best_lpr_model.pth`

### CCPD Filename Format
CCPD filenames encode metadata: `025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg`
- Part 3 (`154&383_386&473`): Bounding box (xmin&ymin_xmax&ymax)
- Part 4 (`386&473_177&454_154&383_363&402`): 4 corners (bottom-right, bottom-left, top-left, top-right order in CCPD)
- Part 5: Character indices into PROVINCES/ALPHABETS/ADS arrays

### Corner Point Ordering
- **CCPD format**: [bottom-right, bottom-left, top-left, top-right]
- **Dataset/LPR format**: [top-left, top-right, bottom-right, bottom-left]
- Dataset parsing handles conversion (indices `[2, 3, 0, 1]`)

### Dependency Constraints
- `opencv-python>=4.10.0.84,<4.12.0` (see pyproject.toml)
- `numpy<2` (compatibility with current opencv version)
