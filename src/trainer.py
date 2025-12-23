# =================================================================
# PURPOSE: Defines the Trainer class for managing the model training
#          and validation processes using PyTorch.
# =================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .model_builder import CatDogClassifier # Import our custom CNN

class Trainer:
    """
    Manages the training and validation loops for the CatDogClassifier model.
    """
    def __init__(self, model: CatDogClassifier, learning_rate: float):
        
        # 1. Device Selection: Selects GPU (cuda) if available, otherwise CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 2. Model Setup: Move the model's parameters to the chosen device.
        self.model = model.to(self.device)

        # 3. Loss Function: Cross Entropy Loss for classification.
        self.criterion = nn.CrossEntropyLoss()

        # 4. Optimizer: Adam optimization algorithm.
        self.optimizer = optim.Adam(
            params=self.model.parameters(), 
            lr=learning_rate
        )

    def train_epoch(self, train_loader: DataLoader):
        """
        Runs one full pass over the training dataset.
        Applies Forward Pass, Backpropagation, and weight updates.
        """
        # Set the model to training mode (important for layers like Dropout/BatchNorm)
        self.model.train() 
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for images, labels in train_loader:
            
            # A. Move Data to Device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # B. Zero the Gradients (Critical PyTorch step)
            self.optimizer.zero_grad()

            # C. Forward Pass: Get predictions (logits)
            outputs = self.model(images)

            # D. Calculate Loss: Measure the error
            loss = self.criterion(outputs, labels)
            
            # E. Backward Pass: Calculate gradients for all parameters
            loss.backward()

            # F. Optimizer Step: Update all model weights
            self.optimizer.step()

            # --- Tracking Metrics ---
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) 
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def validate_epoch(self, validation_loader: DataLoader):
        """
        Runs one full pass over the validation dataset to assess performance.
        NO backpropagation or weight updates occur here.
        """
        # 1. Set the model to evaluation mode
        self.model.eval() 
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # 2. Disable gradient calculation (Saves memory and speeds up validation)
        with torch.no_grad():
            for images, labels in validation_loader:
                
                # A. Move Data to Device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # B. Forward Pass
                outputs = self.model(images)

                # C. Calculate Loss
                loss = self.criterion(outputs, labels)
                
                # --- Tracking Metrics ---
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(validation_loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy