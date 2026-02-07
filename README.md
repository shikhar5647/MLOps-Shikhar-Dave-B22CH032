# CNN Training on CIFAR-10 with Custom DataLoader & Gradient Visualization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive deep learning project implementing CNN-based image classification on CIFAR-10 with custom data loading, computational complexity analysis, gradient flow visualization, and experiment tracking.

---

## 🎯 Overview

This project demonstrates a complete machine learning pipeline for image classification using Convolutional Neural Networks (CNNs). It includes:

- **Custom DataLoader** for CIFAR-10 with flexible augmentation
- **Pre-trained ResNet18** fine-tuned for CIFAR-10
- **FLOPs Analysis** using multiple tools (THOP, PTFLOPs, TorchInfo)
- **Gradient Flow Visualization** to monitor training health
- **Weights & Biases Integration** for comprehensive experiment tracking

### Key Achievements

- ✅ **Custom DataLoader** with augmentation pipeline
- ✅ **Stable Gradient Flow** (no vanishing/exploding)
- ✅ **30 Epoch Training** in ~15-20 minutes (GPU : A6000)

---

## ✨ Features

### 1. Custom DataLoader Implementation
- `CustomCIFAR10Dataset` class extending PyTorch's Dataset
- Flexible data augmentation pipeline
- Configurable transformations for train/test splits
- Efficient batch loading with multi-processing

### 2. Computational Complexity Analysis
- **FLOPs Counting** using three verification methods
- **Parameter Analysis** 
- **Memory Footprint** estimation
- **Layer-wise Complexity** breakdown

### 3. Gradient & Weight Tracking
- Real-time gradient statistics (mean, std, norm)
- Weight evolution monitoring
- Gradient flow visualization every N epochs
- Detection of vanishing/exploding gradients

### 4. Experiment Tracking (WandB)
- Training/validation metrics logging
- Loss curves and accuracy plots
- Gradient statistics heatmaps
- Confusion matrix and per-class accuracy
- Model architecture visualization

---

## Project Structure

```
cifar10-cnn-project/
│
├── cifar10_cnn_complete_assignment.ipynb  # Main notebook
├── README.md                              
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works)
- 8GB RAM minimum (16GB recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/shikhar5647/MLOps-Shikhar-Dave-B22CH032.git
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n cifar10 python=3.8
conda activate cifar10
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
wandb>=0.15.0
thop>=0.1.1
ptflops>=0.7.0
torchinfo>=1.8.0
scikit-learn>=1.3.0
pillow>=10.0.0
```

### Step 4: Setup Weights & Biases (Optional but Recommended)

```bash
# Login to WandB
wandb login

# Or set API key directly
export WANDB_API_KEY=your_api_key_here
```

---

## Quick Start

### Option 1: Google Colab (Easiest)

1. Upload `.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells: Runtime → Run all
4. Login to WandB when prompted

### Option 2: Local Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open .ipynb
# Run all cells
```

### Option 3: Python Script

```python
# Run the training pipeline
python train.py --epochs 30 --batch-size 128 --lr 0.001
```

---

## Methodology

### 1. Dataset Preparation

#### CIFAR-10 Dataset
- **Size:** 60,000 images (50,000 train, 10,000 test)
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Resolution:** 32×32 pixels, RGB
- **Split:** 80% train, 20% validation from training set

#### Custom DataLoader Implementation

```python
class CustomCIFAR10Dataset(Dataset):
    """
    Custom Dataset wrapper for CIFAR-10 with flexible transformations.
    
    Features:
    - Wraps torchvision.datasets.CIFAR10
    - Supports custom augmentation pipelines
    - Easy modification of data loading logic
    """
    
    def __init__(self, root, train=True, transform=None, download=True):
        self.cifar10 = datasets.CIFAR10(root, train, download, transform=None)
        self.transform = transform
        
    def __len__(self):
        return len(self.cifar10)
    
    def __getitem__(self, idx):
        image, label = self.cifar10[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
```

#### Data Augmentation Strategy

**Training Transforms:**
- Random Crop (32×32, padding=4)
- Random Horizontal Flip (p=0.5)
- Normalization (mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

**Test Transforms:**
- Normalization only (no augmentation)

**Rationale:**
- **Random Crop:** Introduces translation invariance
- **Horizontal Flip:** Simulates different viewpoints
- **Normalization:** Centers data around zero for faster convergence

---

### Model Architecture

#### ResNet18 (Pre-trained)

**Why ResNet18?**
- ✅ **Residual Connections:** Prevent vanishing gradients
- ✅ **Transfer Learning:** Pre-trained on ImageNet
- ✅ **Efficiency:** Good accuracy-to-FLOPs ratio
- ✅ **Proven Performance:** Industry-standard architecture

**Architecture Overview:**
```
Input (3×32×32)
    ↓
Conv1 (7×7, stride=2)
    ↓
MaxPool (3×3, stride=2)
    ↓
Residual Block 1 (64 filters) ×2
    ↓
Residual Block 2 (128 filters) ×2
    ↓
Residual Block 3 (256 filters) ×2
    ↓
Residual Block 4 (512 filters) ×2
    ↓
Global Average Pooling
    ↓
Fully Connected (10 classes)
    ↓
Output (10-way classification)
```

**Modifications for CIFAR-10:**
- Final FC layer: 512 → 10 neurons
- All other layers frozen initially (optional)

---

### Training Configuration

#### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 30 | Sufficient for convergence |
| **Batch Size** | 128 | Balance between speed and stability |
| **Learning Rate** | 0.001 | Adam default, works well with fine-tuning |
| **Optimizer** | Adam | Adaptive learning rates per parameter |
| **Loss Function** | CrossEntropyLoss | Standard for multi-class classification |
| **Scheduler** | ReduceLROnPlateau | Adaptive LR based on validation loss |
| **Gradient Clipping** | max_norm=1.0 | Prevents exploding gradients |

#### Optimization Strategy

**Adam Optimizer:**
- Combines momentum and RMSprop
- Adaptive learning rates
- Default β₁=0.9, β₂=0.999, ε=1e-8

**Learning Rate Scheduling:**
```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',        # Minimize validation loss
    factor=0.5,        # Reduce LR by 50%
    patience=3,        # Wait 3 epochs before reduction
    verbose=True
)
```

**Gradient Clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Prevents gradient explosions by scaling gradients when norm > 1.0

---

### FLOPs Analysis

#### What are FLOPs?

**FLOP** = **FL**oating **P**oint **OP**eration

FLOPs measure computational cost:
- **Addition:** 1 FLOP
- **Multiplication:** 1 FLOP
- **Multiply-Accumulate (MAC):** 2 FLOPs

#### Calculation Methods

**1. THOP (Torch-Optimizer)**
```python
from thop import profile
flops, params = profile(model, inputs=(input_tensor,))
```

**2. PTFLOPs**
```python
from ptflops import get_model_complexity_info
macs, params = get_model_complexity_info(model, (3, 32, 32))
```

**3. TorchInfo**
```python
from torchinfo import summary
summary(model, input_size=(1, 3, 32, 32))
```

---

### Gradient Flow & Weight Tracking

#### Why Monitor Gradients?

**Problems Detected:**
- **Vanishing Gradients:** Gradients → 0 (network doesn't learn)
- **Exploding Gradients:** Gradients → ∞ (training instability)
- **Dead Neurons:** Gradients consistently zero
- **Poor Initialization:** Uneven gradient distribution

#### GradientWeightTracker Implementation

```python
class GradientWeightTracker:
    """
    Comprehensive gradient and weight monitoring.
    
    Tracks:
    - Gradient statistics (mean, std, norm, min, max)
    - Weight statistics (mean, std, norm)
    - Weight update magnitudes
    - Layer-wise gradient flow
    """
    
    def update(self):
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Gradient metrics
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                
                # Weight metrics
                weight_norm = param.data.norm().item()
                weight_update = (param.data - prev_weights[name]).norm().item()
                
                # Store for visualization
                self.metrics[name]['grad_norm'].append(grad_norm)
                # ... (other metrics)
```

#### Metrics Tracked

**Per Layer:**
- **Gradient Norm (L2):** `||∇w|| = sqrt(Σ grad²)`
- **Gradient Mean:** Average gradient value
- **Gradient Std:** Spread of gradient values
- **Weight Norm:** `||w|| = sqrt(Σ w²)`
- **Weight Update:** `||w_new - w_old||`

**Healthy Ranges:**
- Gradient Norm: 1e-7 to 100
- Gradient Mean: Close to 0 (balanced updates)
- Weight Updates: Decreasing over time (convergence)

---

### Weights & Biases Integration

#### What is WandB?

Weights & Biases is an MLOps platform for:
- **Experiment Tracking:** Log metrics, hyperparameters
- **Visualization:** Interactive plots and dashboards
- **Model Versioning:** Save and compare models
- **Collaboration:** Share results with team

#### Logged Metrics

**Training Metrics:**
- Loss (per batch and epoch)
- Accuracy (train, validation, test)
- Learning rate changes

**Gradient Metrics:**
- Layer-wise gradient norms
- Gradient mean and std
- Weight update magnitudes

**Model Metrics:**
- FLOPs and parameter count
- Memory usage
- Inference time

**Visualizations:**
- Training curves
- Gradient flow plots
- Confusion matrix
- Per-class accuracy
- Sample predictions

#### Setup WandB

```python
import wandb

# Initialize
wandb.init(
    project="cifar10-cnn-assignment",
    name="resnet18-experiment",
    config={
        "epochs": 30,
        "batch_size": 128,
        "learning_rate": 0.001,
        "model": "ResNet18"
    }
)

# Log metrics
wandb.log({"train/loss": loss, "train/acc": accuracy})

# Log plots
wandb.log({"gradient_flow": wandb.Image(fig)})

# Finish
wandb.finish()
```

---


### Software Stack

```
Python 3.8+
├── PyTorch 2.0+
├── torchvision 0.15+
├── CUDA 11.8+ (for GPU)
├── cuDNN 8.0+
└── WandB 0.15+
```

### Reproducibility

**Set Random Seeds:**
```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Fixed Configuration:**
- Batch size: 128
- Learning rate: 0.001
- Data augmentation: Fixed random seed
- Model initialization: Pre-trained weights

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce batch size
BATCH_SIZE = 64  # or 32

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache
torch.cuda.empty_cache()
```

#### 2. WandB Login Issues

**Error:**
```
wandb: ERROR Unable to authenticate
```

**Solutions:**
```bash
# Re-login
wandb login --relogin

# Use API key directly
export WANDB_API_KEY=your_key
```

#### 3. Slow Data Loading

**Issue:** Training stuck at data loading

**Solutions:**
```python
# Increase num_workers
train_loader = DataLoader(
    dataset,
    num_workers=4,  # Increase this
    pin_memory=True
)

# Use persistent_workers (PyTorch 1.8+)
train_loader = DataLoader(
    dataset,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

#### 4. NaN Loss

**Issue:** Loss becomes NaN during training

**Solutions:**
```python
# Check for NaN in data
assert not torch.isnan(images).any()

# Reduce learning rate
optimizer = Adam(model.parameters(), lr=0.0001)

# Gradient clipping (already implemented)
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Future Improvements

### Short-term

- [ ] **Advanced Augmentation:** CutOut, MixUp, AutoAugment
- [ ] **Mixed Precision Training:** FP16 for 2x speedup
- [ ] **Gradient Accumulation:** Simulate larger batch sizes
- [ ] **Learning Rate Finder:** Optimal LR discovery

### Medium-term

- [ ] **Architecture Search:** EfficientNet, Vision Transformer
- [ ] **Ensemble Methods:** Combine multiple models
- [ ] **Knowledge Distillation:** Student-teacher training
- [ ] **Pruning & Quantization:** Model compression

### Long-term

- [ ] **Multi-GPU Training:** Distributed Data Parallel
- [ ] **AutoML Integration:** Hyperparameter optimization
- [ ] **Production Deployment:** ONNX, TorchScript export
- [ ] **Real-time Inference:** Edge device deployment

---

## Usage Examples

### Basic Training

```python
from train import train_model

# Train with default settings
model, history = train_model(
    epochs=30,
    batch_size=128,
    learning_rate=0.001
)
```

### Custom Configuration

```python
# Advanced configuration
config = {
    'epochs': 50,
    'batch_size': 256,
    'learning_rate': 0.0005,
    'optimizer': 'sgd',
    'scheduler': 'cosine',
    'augmentation': 'autoaugment'
}

model, history = train_model(**config)
```

### Inference

```python
from inference import predict

# Single image prediction
image = load_image('test_image.png')
prediction = predict(model, image)
print(f"Predicted: {prediction['class']}")
print(f"Confidence: {prediction['confidence']:.2%}")

# Batch prediction
images = load_batch('test_images/')
predictions = predict_batch(model, images)
```

### Evaluation

```python
from evaluate import evaluate_model

# Evaluate on test set
metrics = evaluate_model(model, test_loader)
print(f"Test Accuracy: {metrics['accuracy']:.2%}")
print(f"Top-5 Accuracy: {metrics['top5_acc']:.2%}")

# Per-class metrics
print(metrics['classification_report'])
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and setup
git clone https://github.com/shikhar5647/MLOps-Shikhar-Dave-B22CH032.git
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black .
flake8 .
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CIFAR-10 Dataset:** [Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html)
- **ResNet Architecture:** [He et al., 2015](https://arxiv.org/abs/1512.03385)
- **PyTorch Team:** For the excellent deep learning framework
- **Weights & Biases:** For experiment tracking tools
- **THOP, PTFLOPs, TorchInfo:** For FLOPs counting utilities

---

## Contact

**Project Maintainer:** Shikhar Dave
- Email: your.email@example.com
- GitHub: [@shikhar5647](https://github.com/shikhar5647)
---

## 📚 References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
2. Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Technical report.
3. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.
4. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training. ICML.

---


---

<div align="center">
  <strong>⭐ If you find this project helpful, please consider giving it a star! ⭐</strong>
  <br><br>
  Made with ❤️ for the Deep Learning Community
</div>
