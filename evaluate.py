import torch
import os
import warnings
from torchvision import transforms
from PIL import Image, ImageFile
from src.model_builder import CatDogClassifier

# Ignore the specific Pillow warning about truncated files
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

# Tell Pillow it's okay to load these files
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_predictions_from_folder(folder_path, model, device):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    results = {"Cat": 0, "Dog": 0}
    
    print(f"Processing {len(files)} images in {folder_path}...")

    for file_name in files:
        img_path = os.path.join(folder_path, file_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                
                label = "Cat" if predicted.item() == 0 else "Dog"
                results[label] += 1
        except Exception as e:
            print(f"Skipping {file_name} due to error: {e}")

    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'models/cat_dog_cnn.pth'

    # Load Model
    model = CatDogClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Paths to your validation folders
    cat_folder = 'data/validation/cats'
    dog_folder = 'data/validation/dogs'

    print("--- Starting Batch Evaluation ---")
    
    cat_results = get_predictions_from_folder(cat_folder, model, device)
    dog_results = get_predictions_from_folder(dog_folder, model, device)

    print("\n--- Summary Report ---")
    print(f"Images in 'cats' folder predicted as: {cat_results}")
    print(f"Images in 'dogs' folder predicted as: {dog_results}")

if __name__ == "__main__":
    main()