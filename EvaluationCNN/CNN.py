import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder


# Function to train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs = 30):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        # Deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), './EvaluationCNN/CNN.pth')

    print('Best val Acc: {:4f}'.format(best_acc))

def resnet18_fixed(num_classes):
    # Using a pre-trained ResNet model adapted for grayscale images
    model = models.resnet18(pretrained=True)
    # Modify the first convolutional layer
    original_first_layer = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Initialize the new conv1 layer by averaging the weights of the original RGB channels
    with torch.no_grad():
        model.conv1.weight[:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)
    # Adjust the fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    return model

if __name__ =='__main__':

    # Set device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to the image folders
    data_dir = './EvaluationCNN/Image_Evaluation'

    # Image transformations for grayscale images
    train_transform = transforms.Compose([
        transforms.Resize((200, 200)),  # Resizing the image
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Normalization for grayscale
    ])

    val_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # Load datasets
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform) 

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = resnet18_fixed(2)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    train_model(model, criterion, optimizer, scheduler)

