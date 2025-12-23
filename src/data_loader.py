# =================================================================
# FILE: src/data_loader.py
# PURPOSE: Handles file path indexing, image transformations, 
#          and creating PyTorch DataLoader objects.
# =================================================================

import os
import glob
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- Data Indexing ---

def gather_file_paths(data_dir: str) -> List[Tuple[str, int]]:
    """
    Gathers all image file paths and assigns class labels (0 for cat, 1 for dog).
    Expected structure: data_dir/cats/*.jpg, data_dir/dogs/*.jpg
    """
    data_list = []
    
    # Label 0: Cats
    cat_paths = glob.glob(os.path.join(data_dir, 'cats', '*.jpg'))
    cat_paths.extend(glob.glob(os.path.join(data_dir, 'cats', '*.png')))
    data_list.extend([(path, 0) for path in cat_paths])
    
    # Label 1: Dogs
    dog_paths = glob.glob(os.path.join(data_dir, 'dogs', '*.jpg'))
    dog_paths.extend(glob.glob(os.path.join(data_dir, 'dogs', '*.png')))
    data_list.extend([(path, 1) for path in dog_paths])
    
    return data_list

# --- Custom Dataset ---

class CatDogDataset(Dataset):
    """
    A custom PyTorch Dataset to load images and apply transformations.
    """
    def __init__(self, data_list: List[Tuple[str, int]], transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 1. Get file path and label
        img_path, label = self.data_list[idx]
        
        # 2. Load image (using PIL to handle different image types)
        image = Image.open(img_path).convert('RGB') # Convert to RGB to ensure 3 channels
        
        # 3. Apply transformations
        if self.transform:
            image = self.transform(image)
            
        # 4. Return the processed tensor and label
        return image, label

# --- Transformation Pipeline (Updated with Augmentation) ---

def get_transform_pipeline(is_training: bool):
    """
    Defines the sequence of transformations for images.
    Includes aggressive augmentation for the training set to fight overfitting.
    """
    
    # Configuration
    IMAGE_SIZE = 128
    # Standard normalization constants (from ImageNet, a good starting point)
    NORM_MEAN = (0.485, 0.456, 0.406) 
    NORM_STD = (0.229, 0.224, 0.225) 
    
    transform_list = []
    
    # Conditional Augmentation: ONLY for training
    if is_training:
        # 2. Aggressive Data Augmentation
        transform_list.extend([
            # Randomly flip the image horizontally
            transforms.RandomHorizontalFlip(),
            
            # Randomly rotate the image up to 15 degrees
            transforms.RandomRotation(15), 
            
            # Randomly adjust brightness and contrast slightly
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            
            # Randomly crop and resize (forces learning non-centered features)
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0))
        ])
    else:
        # Standard processing for validation (non-random, consistent view)
        transform_list.extend([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
        ])
        
    # 3. Standard Conversion and Normalization (Applies to ALL data)
    transform_list.extend([
        # Convert image to Tensor (0-1 range)
        transforms.ToTensor(),
        # Normalize the tensor using standard means and std devs
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
        
    return transforms.Compose(transform_list)

# --- DataLoader Creation ---

def create_data_loaders(train_list: List, validation_list: List, batch_size: int):
    """
    Creates the Dataset and DataLoader instances for training and validation.
    """
    # 1. Get Transformation Pipelines
    train_transform = get_transform_pipeline(is_training=True)
    val_transform = get_transform_pipeline(is_training=False)
    
    # 2. Create Dataset Instances
    train_dataset = CatDogDataset(train_list, transform=train_transform)
    validation_dataset = CatDogDataset(validation_list, transform=val_transform)
    
    # 3. Define DataLoader Parameters (Common best practices)
    num_workers = min(os.cpu_count(), 4) # Use up to 4 parallel workers for speed
    pin_memory = torch.cuda.is_available() # Pin memory if a GPU is available
    
    # 4. Create DataLoader Instances
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, # MUST be True for training
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=batch_size, 
        shuffle=False, # MUST be False for consistent validation
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Data Loaders created with Batch Size: {batch_size}, Workers: {num_workers}")
    return train_loader, validation_loader