import torch
from torchvision import transforms
from PIL import Image
import os
from src.model_builder import CatDogClassifier

# --- 1. Separate Loading Logic ---
def load_trained_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# --- 2. Clean Prediction Logic ---
def predict_image(model, device, image_path):
    # Match the training/validation transforms exactly
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    try:
        # Load and verify image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
        cat_prob = probabilities[0][0].item()
        dog_prob = probabilities[0][1].item()

        # Decide winner
        if cat_prob > dog_prob:
            label, score = "Cat", cat_prob
        else:
            label, score = "Dog", dog_prob

        return label, score

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

if __name__ == "__main__":
    MODEL_PATH = 'models/cat_dog_cnn.pth'
    TEST_IMAGE = 'test_image.jpg'

    if not os.path.exists(TEST_IMAGE):
        print(f"Image not found: {TEST_IMAGE}")
    else:
        # Load model ONCE
        print("Model Loading...")
        my_model, my_device = load_trained_model(MODEL_PATH)

        # Predict
        result, confidence = predict_image(my_model, my_device, TEST_IMAGE)

        if result:
            print("-" * 30)
            print(f"FINAL PREDICTION: {result}")
            print(f"CONFIDENCE: {confidence*100:.2f}%")
            
            # Improvement 3: Warning for low confidence
            if confidence < 0.70:
                print("Warning: Low confidence. The image might be ambiguous.")
            print("-" * 30)