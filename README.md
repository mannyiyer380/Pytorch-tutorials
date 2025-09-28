# PyTorch Tutorials

A comprehensive collection of PyTorch tutorials covering fundamental concepts to advanced techniques, all with **device-agnostic code** that works seamlessly on CPU, CUDA (NVIDIA GPUs), and MPS (Apple Silicon).

## 🎯 Overview

This repository provides hands-on, practical tutorials for learning PyTorch from scratch. Each tutorial includes:

- **Complete Jupyter notebooks** with detailed explanations
- **Device-agnostic implementation** (CPU/CUDA/MPS support)
- **Real-world datasets** and practical examples
- **Best practices** and common pitfall guidance
- **Progressive difficulty** from basics to advanced topics

## 🚀 Quick Start

### Prerequisites

```bash
# Core requirements
pip install torch torchvision jupyter numpy matplotlib seaborn pandas scikit-learn

# Optional but recommended
pip install pillow tqdm
```

### Repository Structure

```
Pytorch_tutorials/
├── README.md                    # This file
├── 01_basics/                   # PyTorch fundamentals
│   ├── README.md
│   └── pytorch_basics.ipynb
├── 02_regression/               # Linear regression
│   ├── README.md
│   └── linear_regression.ipynb
├── 03_classification/           # Classification models  
│   ├── README.md
│   └── classification.ipynb
└── 04_cnn/                      # Convolutional Neural Networks
    ├── README.md
    └── cnn_tutorial.ipynb
```

## 📚 Tutorials

### [01 - PyTorch Basics](01_basics/)
**Prerequisites**: None  
**Duration**: ~1 hour

Learn the fundamentals of PyTorch:
- Tensor operations and automatic differentiation
- Device-agnostic programming (CPU/GPU/MPS)
- Basic neural network components
- Gradient computation and backpropagation

**Key Topics**: Tensors, autograd, device management, basic training loop

---

### [02 - Linear Regression](02_regression/)
**Prerequisites**: 01_basics  
**Duration**: ~1.5 hours

Build your first machine learning models:
- Linear regression with PyTorch's nn.Module
- Training and validation loops
- Real-world datasets (California Housing)
- Optimizer comparison (SGD, Adam, RMSprop, AdamW)

**Key Topics**: nn.Module, optimizers, loss functions, model evaluation

---

### [03 - Classification](03_classification/)
**Prerequisites**: 01_basics, 02_regression  
**Duration**: ~2 hours

Master classification tasks:
- Binary and multi-class classification
- Neural network architectures with dropout
- Comprehensive evaluation metrics
- Real datasets (Iris, synthetic data)

**Key Topics**: Classification metrics, confusion matrices, model interpretation

---

### [04 - Convolutional Neural Networks](04_cnn/)
**Prerequisites**: All previous tutorials  
**Duration**: ~2.5 hours

Deep dive into computer vision:
- CNN architectures for image classification
- MNIST and CIFAR-10 datasets
- Modern techniques (batch normalization, data augmentation)
- Performance optimization and visualization

**Key Topics**: Conv2d, pooling, batch normalization, data augmentation

---

## 🔧 Device Support

All tutorials include **device-agnostic code** that automatically detects and uses the best available device:

```python
# Automatic device detection
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Apple Silicon)")
else:
    device = torch.device('cpu')
    print("Using CPU")

# All tensors and models automatically use the detected device
model = MyModel().to(device)
data = data.to(device)
```

### Supported Devices
- **CPU**: Works on all systems
- **CUDA**: NVIDIA GPUs with CUDA support
- **MPS**: Apple Silicon (M1/M2/M3) Macs with macOS 12.3+

## 📊 What You'll Build

By completing these tutorials, you'll have built:

1. **Tensor manipulation** and gradient computation systems
2. **Linear regression models** for continuous prediction
3. **Classification networks** for categorical prediction  
4. **Convolutional Neural Networks** for image recognition
5. **Complete training pipelines** with evaluation and visualization

## 🛠️ Installation & Setup

### Option 1: Conda Environment (Recommended)
```bash
# Create new environment
conda create -n pytorch-tutorials python=3.9
conda activate pytorch-tutorials

# Install PyTorch (choose based on your system)
# For CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For MPS (Apple Silicon)
conda install pytorch torchvision -c pytorch

# For CPU only
conda install pytorch torchvision cpuonly -c pytorch

# Install additional dependencies
pip install jupyter matplotlib seaborn pandas scikit-learn pillow
```

### Option 2: pip Installation
```bash
# Install PyTorch (visit pytorch.org for system-specific commands)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA
# OR
pip install torch torchvision  # CPU/MPS

# Install additional dependencies
pip install jupyter numpy matplotlib seaborn pandas scikit-learn pillow
```

### Option 3: Google Colab
All notebooks work directly in Google Colab with GPU support:
1. Upload notebooks to Google Drive
2. Open with Google Colab
3. Enable GPU: Runtime → Change runtime type → GPU

## 🎓 Learning Path

### Beginner Path (New to Deep Learning)
1. Start with **01_basics** - Learn PyTorch fundamentals
2. Move to **02_regression** - Understand training loops
3. Continue with **03_classification** - Master evaluation metrics
4. Finish with **04_cnn** - Explore computer vision

### Intermediate Path (Some ML Experience)
1. Skim **01_basics** for PyTorch syntax
2. Focus on **02_regression** for PyTorch workflows
3. Study **03_classification** for advanced techniques
4. Deep dive into **04_cnn** for modern architectures

### Advanced Path (Experienced Practitioners)
- Use tutorials as reference for PyTorch best practices
- Focus on device-agnostic implementation patterns
- Study modern techniques (batch norm, data augmentation)
- Adapt architectures for your specific use cases

## 🔍 Key Features

### Device Agnostic Design
- **Automatic device detection** and optimal performance
- **Cross-platform compatibility** (Windows, macOS, Linux)
- **GPU acceleration** when available, CPU fallback guaranteed

### Practical Focus
- **Real datasets** (MNIST, CIFAR-10, California Housing, Iris)
- **Production-ready code** with proper error handling
- **Best practices** for model development and evaluation

### Comprehensive Coverage
- **Mathematical foundations** with code implementation
- **Visualization** of training progress and model performance
- **Common pitfalls** and how to avoid them

### Educational Value
- **Step-by-step explanations** with code comments
- **Progressive complexity** building on previous concepts
- **Hands-on exercises** for reinforcement learning

## 📈 Performance & Optimization

### Training Speed Comparison (Approximate)
| Dataset | Device | Training Time | Speedup |
|---------|--------|---------------|---------|
| MNIST | CPU | ~5 min | 1x |
| MNIST | CUDA | ~30 sec | 10x |
| MNIST | MPS | ~45 sec | 7x |
| CIFAR-10 | CPU | ~45 min | 1x |
| CIFAR-10 | CUDA | ~3 min | 15x |
| CIFAR-10 | MPS | ~5 min | 9x |

### Memory Usage Tips
- **Batch size**: Adjust based on available GPU memory
- **Model size**: Use mixed precision for larger models
- **Data loading**: Use `num_workers` for faster data loading

## 🧪 Testing Your Setup

Run this quick test to verify your installation:

```python
import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# Test device detection
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.device_count()} devices")
    print(f"Current device: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():
    print("MPS (Apple Silicon) available")
else:
    print("Using CPU")

# Test tensor creation
x = torch.randn(3, 4)
print(f"Created tensor: {x.shape}")
print("✅ Setup successful!")
```

## 🤝 Contributing

We welcome contributions! Here's how to help:

1. **Report bugs** or suggest improvements via issues
2. **Add new tutorials** following the established format
3. **Improve documentation** and add more examples
4. **Optimize performance** for different hardware configurations

### Development Setup
```bash
git clone <your-fork>
cd pytorch-tutorials
pip install -r requirements.txt
jupyter notebook
```

## 📝 Requirements

### Minimum Requirements
- **Python**: 3.7+
- **PyTorch**: 1.12+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB for datasets and notebooks

### Recommended Specifications
- **Python**: 3.9+
- **PyTorch**: Latest stable version
- **GPU**: NVIDIA RTX series or Apple Silicon
- **RAM**: 16GB
- **Storage**: 10GB (for experimentation)

## 🆘 Troubleshooting

### Common Issues

**Issue**: `RuntimeError: No CUDA devices available`  
**Solution**: Ensure CUDA drivers are installed, or use CPU/MPS device

**Issue**: `ModuleNotFoundError: No module named 'torch'`  
**Solution**: Install PyTorch using the correct command for your system

**Issue**: Slow training on GPU  
**Solution**: Increase batch size, check GPU memory usage

**Issue**: Out of memory errors  
**Solution**: Reduce batch size or model complexity

### Getting Help
1. Check tutorial README files for specific guidance
2. Review notebook comments and documentation
3. Open an issue with error details and system information
4. Join PyTorch community forums for additional support

## 📚 Additional Resources

### Official Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Torchvision Documentation](https://pytorch.org/vision/)

### Recommended Reading
- [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch-book/)
- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [Papers with Code](https://paperswithcode.com/lib/pytorch)

### Community
- [PyTorch Forums](https://discuss.pytorch.org/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙋‍♀️ Support

If you find these tutorials helpful:
- ⭐ **Star this repository** to show your support
- 🔄 **Share** with others learning PyTorch
- 🐛 **Report issues** to help improve the tutorials
- 💡 **Suggest improvements** for better learning experience

---

**Happy Learning! 🚀**

Start your PyTorch journey with [01_basics](01_basics/) and build your way up to advanced computer vision with [04_cnn](04_cnn/). Each tutorial builds on the previous one, so take your time and enjoy the process of learning one of the most powerful deep learning frameworks available today!