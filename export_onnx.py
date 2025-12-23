import torch
import torch.onnx
from src.model_builder import CatDogClassifier

def export_model():
    # 1. Load your trained model
    model = CatDogClassifier()
    model.load_state_dict(torch.load('models/cat_dog_cnn.pth', map_location='cpu'))
    model.eval()

    # 2. Create 'dummy' input
    dummy_input = torch.randn(1, 3, 128, 128)

    # 3. Export with modern opset
    onnx_path = "models/cat_dog_model.onnx"
    
    print("Exporting model to ONNX format...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=18, # Updated to version 18 for compatibility with Python 3.13
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output']
    )

    print(f"Success! Model converted to ONNX at: {onnx_path}")

if __name__ == "__main__":
    export_model()