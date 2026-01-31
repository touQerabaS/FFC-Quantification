# RustNet: Filiform Corrosion Detection and Quantification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust deep learning framework for automatic detection, segmentation, and quantification of Filiform Corrosion (FFC) using advanced image processing and segmentation-based algorithms.

![Framework Overview](media/Fig 1.tif)

## Overview

Filiform corrosion (FFC) is an insidious degradation phenomenon characterized by thread-like patterns that severely damage metallic infrastructures under protective organic coatings. This repository provides **RustNet**, a state-of-the-art model that:

- **Detects** FFC filaments with high accuracy
- **Segments** corrosion regions using multi-layer edge attention networks
- **Quantifies** filament lengths in real-world measurements (millimeters)
- **Tracks** corrosion progression over time

## Key Features

- 🎯 **High Accuracy**: Achieves 87.1% accuracy, 0.90 Dice coefficient, and 0.83 IoU
- 📏 **Precise Measurement**: Filament length measurement with <1% error margin
- ⚡ **Efficient Processing**: Optimized for real-time corrosion assessment
- 🔧 **Modular Design**: Easy to extend and customize for different applications

## Architecture

RustNet consists of three main components:

1. **Feature Encoder**: Extracts multi-scale features using convolutional and down-sampling operations
2. **Feature Decoder**: Integrates features from encoding and decoding layers through stacked decoding blocks
3. **Rust Module**: Contains two submodules:
   - **RFE (Rust Feature Extraction)**: Captures edge information and generates attention maps
   - **MAN (Multilayer Attention Network)**: Filters edge information with different attention maps

```
Input Image → Feature Encoder → Rust Module (RFE + MAN) → Feature Decoder → Segmentation Mask
                                                                         ↓
                                                              Skeletonization → Length Measurement
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/touQerabaS/rustnet-ffc.git
cd rustnet-ffc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.5.0
scikit-image>=0.19.0
networkx>=2.6.0
matplotlib>=3.5.0
pillow>=9.0.0
tqdm>=4.62.0
```

## Dataset

### Structure

```
data/
├── train/
│   ├── images/          # Original FFC images (1937 samples)
│   └── masks/           # Binary segmentation masks
├── val/
│   ├── images/          # Validation images (424 samples)
│   └── masks/
└── test/
    ├── images/          # Test images (424 samples)
    └── masks/
```

### Data Preparation

Images should be resized to **512 × 512 pixels**. Binary masks should have:
- **White (255)**: Corrosion filament regions
- **Black (0)**: Background

### Generating FFC Samples (Laboratory Protocol)

1. Create a 10mm scratch on coated ZAM-steel substrate using a scalpel blade
2. Inject 3 μL of dilute acetic acid solution (pH ~3) into the scratch
3. Place specimens in humidity desiccator with Na₂SO₄·10H₂O solution (~95% RH)
4. Monitor progression using optical microscope

## Usage

### Training

```bash
python train.py \
    --data_dir ./data \
    --backbone resnet34 \
    --loss focal \
    --epochs 35 \
    --lr 1e-4 \
    --batch_size 8 \
    --output_dir ./checkpoints
```

### Inference

```bash
python inference.py \
    --input ./test_images \
    --model ./checkpoints/rustnet_best.pth \
    --output ./results
```

### Filament Measurement

```bash
python measure.py \
    --image ./sample.png \
    --model ./checkpoints/rustnet_best.pth \
    --baseline_mm 10.0  # Known baseline length in mm
```

### Python API

```python
from rustnet import RustNet, FilamentMeasurer

# Load model
model = RustNet(backbone='resnet34', pretrained=True)
model.load_state_dict(torch.load('checkpoints/rustnet_best.pth'))
model.eval()

# Segment FFC
image = load_image('sample.png')
mask = model.predict(image)

# Measure filaments
measurer = FilamentMeasurer(baseline_mm=10.0)
results = measurer.measure(mask)

for filament in results:
    print(f"Filament {filament['id']}: {filament['length_mm']:.2f} mm")
```

## Model Configuration

| Model | Backbone | Loss | Accuracy | Dice | IoU |
|-------|----------|------|----------|------|-----|
| U-Net | VGG-19 | Focal | 83% | 0.92 | 0.81 |
| FPN | InceptionV4 | BCE | 83% | 0.91 | 0.82 |
| **RustNet** | **ResNet-34** | **Focal** | **87%** | **0.90** | **0.83** |

## Pipeline Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data           │     │  Segmentation   │     │  Post-          │
│  Preparation    │ ──► │  (RustNet)      │ ──► │  Processing     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Length         │     │  Parameter      │     │  Skeleton       │
│  Measurement    │ ◄── │  Estimation     │ ◄── │  Extraction     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Skeletonization Algorithm

The framework uses medial-axis skeletonization to extract filament structures:

1. **Skeleton Initialization**: Convert binary mask to single-pixel skeleton
2. **Graph Construction**: Build graph G = (V, E) from skeleton pixels
3. **Branch Extraction**: Identify individual branches based on node degrees
4. **Parameter Estimation**: Calculate branch lengths using Euclidean distances

```python
# Branch length calculation
L(B_i) = Σ sqrt((x_j - x_i)² + (y_j - y_i)²)
```

## Results

### Segmentation Performance

- **Training Accuracy**: 87.1%
- **Validation Loss**: 0.16
- **Mask Loss**: 0.1
- **Dice Coefficient**: 0.90
- **IoU Score**: 0.83

### Measurement Accuracy

| Scenario | True Length | Predicted Length | Accuracy |
|----------|-------------|------------------|----------|
| Simple (single filament) | 10.00 mm | 9.87 mm | 98.7% |
| Complex (multiple filaments) | Variable | Variable | 95.5-98.9% |



## Related Work

- [RustNet: Automated Rust Detection](https://link.springer.com/article/10.1007/s43452-025-01312-5) - Original RustNet architecture
- [U-Net](https://arxiv.org/abs/1505.04597) - Convolutional Networks for Biomedical Image Segmentation
- [FPN](https://arxiv.org/abs/1612.03144) - Feature Pyramid Networks for Object Detection



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


<p align="center">
  <i>Developed at Beijing University of Chemical Technology</i>
</p>
