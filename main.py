import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
import cv2
import numpy as np

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cat-Dog Classifier")
        self.setGeometry(100, 100, 800, 600)
        
        # Set up device
        self.device = get_device()
        print(f'Using device: {self.device}')
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create buttons
        self.btn_show_images = QPushButton("1. Show Images")
        self.btn_show_distribution = QPushButton("2. Show Distribution")
        self.btn_show_structure = QPushButton("3. Show Model Structure")
        self.btn_show_comparison = QPushButton("4. Show Comparison")
        self.btn_load_image = QPushButton("Load Image")
        self.btn_inference = QPushButton("5. Inference")
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # Create result label
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to layout
        layout.addWidget(self.btn_show_images)
        layout.addWidget(self.btn_show_distribution)
        layout.addWidget(self.btn_show_structure)
        layout.addWidget(self.btn_show_comparison)
        layout.addWidget(self.btn_load_image)
        layout.addWidget(self.image_label)
        layout.addWidget(self.btn_inference)
        layout.addWidget(self.result_label)
        
        # Connect buttons to functions
        self.btn_show_images.clicked.connect(self.show_images)
        self.btn_show_distribution.clicked.connect(self.show_distribution)
        self.btn_show_structure.clicked.connect(self.show_structure)
        self.btn_show_comparison.clicked.connect(self.show_comparison)
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_inference.clicked.connect(self.run_inference)
        
        # Initialize model
        self.model = None
        self.current_image = None
        self.load_model()
        
    def load_model(self):
        # Create model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
        
        # Load trained weights (using the better model)
        checkpoint = torch.load('model/best_model_FocalLoss.pth', 
                              map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def run_inference(self):
        """Run inference on the loaded image"""
        try:
            if self.current_image is None:
                self.result_label.setText("Please load an image first!")
                return
                
            # Prepare image for inference
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            # Apply transforms
            input_tensor = transform(rgb_image)
            # Add batch dimension and move to device
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_batch)
                predicted_prob = output.item()
                predicted_class = "Cat" if predicted_prob < 0.5 else "Dog"
                confidence = predicted_prob if predicted_class == "Dog" else 1 - predicted_prob
                
            self.result_label.setText(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2%})")
        except Exception as e:
            self.result_label.setText(f"Error during inference: {str(e)}")

    def show_images(self):
        """Display randomly selected cat and dog images from inference dataset"""
        try:
            import os
            import random
            
            # Define inference dataset paths
            base_path = './inference_dataset'
            cat_dir = os.path.join(base_path, 'cat')
            dog_dir = os.path.join(base_path, 'dog')
            
            def get_random_image_path(directory):
                # Get list of all jpg files in directory
                image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if not image_files:
                    raise ValueError(f"No images found in {directory}")
                # Randomly select one image
                random_image = random.choice(image_files)
                return os.path.join(directory, random_image)
            
            def load_and_resize_image(image_path):
                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not load image from {image_path}")
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to 224x224
                img = cv2.resize(img, (224, 224))
                return img
            
            # Get random image paths
            cat_path = get_random_image_path(cat_dir)
            dog_path = get_random_image_path(dog_dir)
            
            # Load and resize images
            cat_img = load_and_resize_image(cat_path)
            dog_img = load_and_resize_image(dog_path)
            
            # Create a combined image
            combined_img = np.hstack((cat_img, dog_img))
            
            # Add titles
            title_space = np.ones((30, combined_img.shape[1], 3), dtype=np.uint8) * 255
            # Add file names to titles
            cat_name = os.path.basename(cat_path)
            dog_name = os.path.basename(dog_path)
            cv2.putText(title_space, f'Cat: {cat_name}', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(title_space, f'Dog: {dog_name}', (224 + 10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Combine with title
            final_img = np.vstack((title_space, combined_img))
            
            # Convert to QImage
            height, width, channel = final_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(final_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Display image
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), 
                                                self.image_label.height(), 
                                                Qt.KeepAspectRatio))
            
            # Update result label with image details
            self.result_label.setText(
                f"Randomly selected images:\n"
                f"Cat: {cat_name}\n"
                f"Dog: {dog_name}\n"
                f"Images resized to 224×224×3 (RGB)"
            )
            
            # Print verification information
            print(f"Selected cat image: {cat_path}")
            print(f"Selected dog image: {dog_path}")
            print(f"Cat image shape: {cat_img.shape}")
            print(f"Dog image shape: {dog_img.shape}")
            
        except Exception as e:
            error_msg = f"Error showing images: {str(e)}"
            self.result_label.setText(error_msg)
            print(f"Error details: {str(e)}")

    def show_distribution(self):
        """Display the class distribution plot"""
        try:
            img = cv2.imread('class_distribution.png')
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), 
                                                       self.image_label.height(), 
                                                       Qt.KeepAspectRatio))
            else:
                self.result_label.setText("Error: Could not load distribution plot")
        except Exception as e:
            self.result_label.setText(f"Error showing distribution: {str(e)}")

    def show_structure(self):
        """Display the model structure"""
        try:
            # Create a text representation of the model structure
            model_structure = str(self.model)
            self.result_label.setText(model_structure)
        except Exception as e:
            self.result_label.setText(f"Error showing model structure: {str(e)}")

    def show_comparison(self):
        """Display the accuracy comparison plot"""
        try:
            img = cv2.imread('accuracy_comparison.png')
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), 
                                                       self.image_label.height(), 
                                                       Qt.KeepAspectRatio))
            else:
                self.result_label.setText("Error: Could not load comparison plot")
        except Exception as e:
            self.result_label.setText(f"Error showing comparison: {str(e)}")

    def load_image(self):
        """Load an image file for inference"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File",
                                                     "", "Images (*.png *.xpm *.jpg *.bmp)")
            if file_name:
                self.current_image = cv2.imread(file_name)
                if self.current_image is not None:
                    # Convert BGR to RGB for display
                    display_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                    height, width, channel = display_image.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(display_image.data, width, height, bytes_per_line, 
                                 QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), 
                                                           self.image_label.height(), 
                                                           Qt.KeepAspectRatio))
                    self.result_label.setText("Image loaded successfully")
                else:
                    self.result_label.setText("Error: Could not load the image")
        except Exception as e:
            self.result_label.setText(f"Error loading image: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()