# FILE: src/model_builder.py

import torch
import torch.nn as nn
import torch.nn.functional as F # Imported as F for common activation functions

class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()

        # --- Feature Extractor (Convolutional Blocks) ---
        # Input: (BatchSize, 3, 128, 128)

        # Block 1: 3 -> 16 channels, MaxPool reduces spatial size to 64x64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 16 -> 32 channels, MaxPool reduces spatial size to 32x32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 32 -> 64 channels, MaxPool reduces spatial size to 16x16
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Classifier Head (Fully Connected Layers) ---

        # The input size to the Linear layer is calculated based on the last block's output:
        # Spatial size: 128 / (2*2*2) = 16 (for height and width)
        # Channels: 64
        self.fc_input_size = 16 * 16 * 64 # = 16384
        
        # The output must be 2 for the two classes (Cat and Dog)
        self.fc1 = nn.Linear(in_features=self.fc_input_size, out_features=2) 

    def forward(self, x):
        # x starts as: (BatchSize, 3, 128, 128)
        
        # Block 1: Conv -> ReLU -> Pool
        x = self.pool1(F.relu(self.conv1(x)))
        # x is now: (BatchSize, 16, 64, 64)
        
        # Block 2: Conv -> ReLU -> Pool
        x = self.pool2(F.relu(self.conv2(x)))
        # x is now: (BatchSize, 32, 32, 32)
        
        # Block 3: Conv -> ReLU -> Pool
        x = self.pool3(F.relu(self.conv3(x)))
        # x is now: (BatchSize, 64, 16, 16)

        # Flatten: Reshape the 3D feature map into a 1D vector for the classifier
        x = x.view(-1, self.fc_input_size)
        
        # Classification
        output = self.fc1(x)
        
        return output