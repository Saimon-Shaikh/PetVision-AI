# =================================================================
# PURPOSE: The main entry point and orchestrator for the Cat vs Dog 
#          classification project. Initializes data, model, and trainer.
# =================================================================

import os
import torch
import torch.nn as nn
import warnings

# Imports from your structured project files
from src.data_loader import gather_file_paths, create_data_loaders
from src.model_builder import CatDogClassifier
from src.trainer import Trainer
from PIL import ImageFile

# Ignore the specific Pillow warning about truncated files
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

# Tell Pillow it's okay to load these files
ImageFile.LOAD_TRUNCATED_IMAGES = True
# --- GLOBAL CONFIGURATION ---
DATA_DIR = 'data' 
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10 # Running for 10 epochs to observe the effect of augmentation

def main():
    print("--- Starting Cat vs Dog Classifier Training ---")
    
    # 1. DATA PREPARATION (Indexing the files)
    try:
        train_list = gather_file_paths(os.path.join(DATA_DIR, 'train'))
        validation_list = gather_file_paths(os.path.join(DATA_DIR, 'validation'))
    except Exception as e:
        print(f"Error during file gathering: {e}")
        print("FATAL: Ensure your 'data' folder and its 'train/cats', 'train/dogs' structure is correct.")
        return

    print(f"Total training images found: {len(train_list)}")
    print(f"Total validation images found: {len(validation_list)}")
    
    # Safety Check
    if len(train_list) < 10 or len(validation_list) < 1:
        print("FATAL: Insufficient data found. Please ensure both train and validation folders have images.")
        return

    # 2. CREATE DATALOADERS
    train_loader, validation_loader = create_data_loaders(
        train_list, 
        validation_list, 
        batch_size=BATCH_SIZE
    )

    # 3. MODEL AND TRAINER SETUP
    # Instantiates the CNN model
    model = CatDogClassifier()
    # Instantiates the Trainer, handling device, loss, and optimizer setup
    trainer = Trainer(model, learning_rate=LEARNING_RATE)

    # 4. MAIN TRAINING LOOP
    print("\nStarting training...\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training Phase: Model learns and updates weights
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # Validation Phase: Model is tested on unseen data
        val_loss, val_acc = trainer.validate_epoch(validation_loader)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
    # 5. SAVE THE TRAINED MODEL
    print("\n--- Training Complete ---")
    
    # Create a directory for saved models if it doesn't exist
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    
    # Define the save path
    model_path = os.path.join(save_dir, 'cat_dog_cnn.pth')
    
    # Save the model's learned weights (the state_dict)
    torch.save(model.state_dict(), model_path)
    
    print(f"Successfully saved model weights to: {model_path}")



if __name__ == "__main__":
    main()