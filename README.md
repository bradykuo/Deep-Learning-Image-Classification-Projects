# Deep Learning Image Classification Projects
This repository contains two deep learning image classification projects: CIFAR10 Classification using VGG19 and Cat-Dog Classification using ResNet50. Both projects feature PyQt5-based GUI interfaces for visualization and inference.

## Prerequisites

### Python Version
- Python 3.7

### Required Libraries
```bash
pip install opencv-contrib-python==3.4.2.17
pip install matplotlib==3.1.1
pip install pyqt5==5.15.1
pip install torch torchvision
pip install tensorflow
pip install numpy
pip install tqdm
pip install torchsummary
```

## Project 1: CIFAR10 Image Classification with VGG19

### Project Structure
- `main.py`: GUI implementation and main application logic
- `train.py`: Model training script
- `training_history.png`: Training metrics visualization
- `vgg19_weights.keras`: Saved model weights
- `vgg19_final.keras`: Final trained model

### Features
1. **Data Visualization**
   - Random display of 9 CIFAR10 training images with labels
   - Support for 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

2. **Model Structure**
   - VGG19 architecture adapted for CIFAR10
   - Input shape: 32x32x3
   - Output: 10 classes

3. **Data Augmentation**
   - RandomRotation
   - RandomResizedCrop
   - RandomHorizontalFlip
   - Real-time visualization of augmentation effects

4. **Training Configuration**
   - Epochs: 30
   - Loss Function: Categorical Crossentropy
   - Optimizer: Adam
   - Batch Size: 32
   - Data Split: Training and Testing sets

## Project 2: Cat-Dog Classification with ResNet50

### Project Structure
```
Hw2_05_StudentID_Name_Version/
├── model/                  # Folder for trained models
│   └── best_model_FocalLoss.pth
├── inference_dataset/      # Test images
│   ├── cat/
│   └── dog/
├── main.py                 # GUI implementation
├── train.py               # Training script
├── class_distribution.png  # Training data visualization
└── accuracy_comparison.png # Model comparison results
```

### Features
1. **Dataset Loading and Visualization**
   - Load and resize images to 224x224x3 (RGB)
   - Display random samples from inference dataset
   - Visualize class distribution in training data

2. **Model Architecture**
   - ResNet50 backbone with transfer learning
   - Modified final layer for binary classification
   - Output: Confidence value (0~1)

3. **Training Implementation**
   - Two training approaches:
     - Binary Cross Entropy Loss
     - Focal Loss (α-balanced)
   - Handles imbalanced dataset
   - Model checkpointing for best performance

4. **Training Configuration**
   - Loss Functions: BCE and Focal Loss
   - Optimizer: Adam
   - Learning Rate: 0.0001
   - Batch Size: 32
   - Image Size: 224x224
   - Normalization: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Common GUI Interface Features

Both applications feature dark-themed GUIs with:
- Left Panel:
  - Load Image button
  - Function buttons for different operations
  - Title and controls
- Right Panel:
  - Image display area
  - Results visualization

## Usage

### Running the Applications
```bash
# For CIFAR10 Classification
python cifar10_main.py

# For Cat-Dog Classification
python catdog_main.py
```

### Training the Models
```bash
# For CIFAR10 Classification
python cifar10_train.py

# For Cat-Dog Classification
python catdog_train.py
```

## Known Issues and Limitations

1. Both applications require significant RAM for model loading
2. GPU acceleration is recommended for optimal performance
3. CIFAR10 project restricts image input size to 32x32
4. Cat-Dog project requires 224x224 input images

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

### CIFAR10 Project
- CIFAR10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- VGG19 architecture: https://pytorch.org/vision/0.12/generated/torchvision.models.vgg19.html
- VGG19_references: https://reurl.cc/rvpRXZ

### Cat-Dog Project
- Deep Residual Learning for Image Recognition (ResNet)
- Focal Loss for Dense Object Detection
- Kaggle Cats and Dogs Dataset
- ResNet50_reference: https://reurl.cc/Ord4z3

## License

This project is available for academic and educational purposes.

## Acknowledgments

- Thanks to Benjamin for the Cat-Dog classification problem formulation
- Thanks to the PyTorch and TensorFlow teams for their excellent frameworks
- Thanks to the original authors of VGG19 and ResNet50 architectures
