import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

# Custom Dataset class for loading images
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets.float())
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

# Function to create and modify ResNet50 model
def create_model(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    model = model.to(device)
    return model

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    best_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                predicted = (outputs.squeeze() >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels.float()).sum().item()
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            model_save_path = os.path.join('model', f'best_model_{criterion.__class__.__name__}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, model_save_path)
    
    return train_losses, val_accuracies

def main():
    # Set device
    device = get_device()
    print(f'Using device: {device}')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = CatDogDataset('Dataset_OpenCvDl_Hw2_Q5/train', transform=transform)
    val_dataset = CatDogDataset('Dataset_OpenCvDl_Hw2_Q5/val', transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, 
                            batch_size=32, 
                            shuffle=True,
                            num_workers=0)  # MPS doesn't support num_workers > 0
    
    val_loader = DataLoader(val_dataset, 
                          batch_size=32, 
                          shuffle=False,
                          num_workers=0)
    
    # Plot class distribution
    labels = [label for _, label in train_dataset.dataset.samples]
    class_counts = np.bincount(labels)
    plt.figure(figsize=(8, 6))
    plt.bar(['Cat', 'Dog'], class_counts)
    plt.title('Class Distribution in Training Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.savefig('class_distribution.png')
    plt.close()
    
    # Calculate alpha for Focal Loss
    total_samples = sum(class_counts)
    cat_frequency = class_counts[0] / total_samples
    alpha = 1 - cat_frequency
    
    # Train with Focal Loss
    print("\nTraining model with Focal Loss...")
    model_focal = create_model(device)
    criterion_focal = FocalLoss(alpha=alpha)
    optimizer_focal = optim.Adam(model_focal.parameters(), lr=0.0001)
    
    losses_focal, acc_focal = train_model(
        model_focal, train_loader, val_loader,
        criterion_focal, optimizer_focal, num_epochs=10, device=device
    )
    
    # Train with BCE Loss
    print("\nTraining model with BCE Loss...")
    model_bce = create_model(device)
    criterion_bce = nn.BCELoss()
    optimizer_bce = optim.Adam(model_bce.parameters(), lr=0.0001)
    
    losses_bce, acc_bce = train_model(
        model_bce, train_loader, val_loader,
        criterion_bce, optimizer_bce, num_epochs=10, device=device
    )
    
    # Plot accuracy comparison
    plt.figure(figsize=(8, 6))
    plt.bar(['Focal Loss', 'BCE Loss'], [max(acc_focal), max(acc_bce)])
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Loss Function')
    plt.ylabel('Validation Accuracy (%)')
    plt.savefig('accuracy_comparison.png')
    plt.close()

if __name__ == '__main__':
    main()