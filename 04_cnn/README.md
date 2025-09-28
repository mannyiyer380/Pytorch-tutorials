# Convolutional Neural Networks (CNNs) with PyTorch

This tutorial covers implementing Convolutional Neural Networks using PyTorch for image classification, including proper CNN architectures, data preprocessing, and advanced techniques.

## Topics Covered

- CNN architecture fundamentals (Conv2d, MaxPool2d, BatchNorm2d)
- Image data preprocessing and augmentation
- MNIST handwritten digit classification
- CIFAR-10 object classification
- Modern CNN techniques (Dropout, Batch Normalization)
- Transfer learning concepts
- Model visualization and interpretation
- Device-agnostic implementation with GPU acceleration

## Prerequisites

- Completion of previous tutorials (01_basics, 02_regression, 03_classification)
- Understanding of convolutional operations
- Basic knowledge of computer vision concepts

## Installation

```bash
pip install torch torchvision jupyter numpy matplotlib seaborn pandas pillow
```

## Contents

- `cnn_tutorial.ipynb`: Complete CNN implementation
- MNIST digit classification (simple CNN)
- CIFAR-10 object classification (deeper CNN)
- Data augmentation techniques
- Model evaluation and visualization
- Performance optimization tips

## Key Learning Objectives

- Build CNN architectures using PyTorch's nn.Conv2d, nn.MaxPool2d
- Implement proper image data preprocessing pipelines
- Use torchvision for datasets and transforms
- Apply data augmentation for better generalization
- Understand CNN layer dimensions and parameter calculations
- Visualize CNN filters and feature maps
- Compare performance with and without modern techniques

## Datasets Used

1. MNIST: 28x28 grayscale handwritten digits (10 classes)
2. CIFAR-10: 32x32 color images of objects (10 classes)

## CNN Techniques Covered

- Convolutional layers with various kernel sizes
- Max pooling for dimensionality reduction
- Batch normalization for training stability
- Dropout for regularization
- Adaptive pooling for flexible input sizes
- Learning rate scheduling

## Next Steps

After completing this tutorial:
- Explore advanced architectures (ResNet, VGG, etc.)
- Learn about transfer learning and pre-trained models
- Study object detection and segmentation
- Investigate generative models (GANs, VAEs)