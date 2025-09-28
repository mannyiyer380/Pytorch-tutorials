# Classification with PyTorch

This tutorial covers implementing classification models using PyTorch, including binary and multi-class classification with proper neural network architectures and evaluation techniques.

## Topics Covered

- Binary and multi-class classification
- Neural network architectures for classification
- Loss functions (CrossEntropyLoss, BCELoss)
- Activation functions (ReLU, Sigmoid, Softmax)
- Model evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
- Data preprocessing and feature engineering
- Handling imbalanced datasets
- Device-agnostic implementation

## Prerequisites

- Completion of `01_basics` and `02_regression` tutorials
- Understanding of classification concepts
- Basic knowledge of machine learning metrics

## Installation

```bash
pip install torch torchvision scikit-learn jupyter numpy matplotlib seaborn pandas
```

## Contents

- `classification.ipynb`: Complete classification implementation
- Binary classification with synthetic data
- Multi-class classification with Iris dataset
- Wine quality classification (real-world example)
- Model evaluation and visualization techniques

## Key Learning Objectives

- Build classification models using PyTorch's nn.Module
- Implement multi-layer neural networks
- Use appropriate loss functions for classification
- Evaluate classification performance with multiple metrics
- Visualize decision boundaries and confusion matrices
- Handle different types of classification problems

## Datasets Used

1. Synthetic binary classification data
2. Iris flower classification (multi-class)
3. Wine quality dataset (multi-class, real-world)

## Next Steps

After completing this tutorial:
- 04_cnn: Convolutional Neural Networks for image classification
- Advanced topics: Regularization, Dropout, Batch Normalization