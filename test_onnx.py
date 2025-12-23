import onnxruntime as ort
import numpy as np
from PIL import Image
import os

def test_prediction(image_path, model_path):
    print(f"--- Testing ONNX Model: {model_path} ---")
    
    # 1. Start the ONNX inference session
    # This automatically detects the .onnx and the accompanying data file
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # 2. Image Pre-processing (Matches the training logic exactly)
    try:
        # Load and convert to RGB
        img = Image.open(image_path).convert('RGB')
        # Resize to the 128x128 expected by your CNN
        img = img.resize((128, 128))
        
        # Convert to numpy array and scale to [0, 1]
        img_data = np.array(img).astype('float32') / 255.0
        
        # Normalize using ImageNet standards (which your trainer used)
        mean = np.array([0.485, 0.456, 0.406], dtype='float32')
        std = np.array([0.229, 0.224, 0.225], dtype='float32')
        img_data = (img_data - mean) / std
        
        # Transpose: Change from (Height, Width, Channel) to (Channel, Height, Width)
        img_data = np.transpose(img_data, (2, 0, 1))
        
        # Add Batch dimension and force float32 to prevent 'Double' errors
        img_data = np.expand_dims(img_data, axis=0).astype('float32')
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # 3. Run the Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_data})

    # 4. Post-processing (Softmax)
    logits = outputs[0]
    # We apply softmax manually to get the confidence percentage
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    prediction = np.argmax(probs)
    confidence = probs[0][prediction] * 100
    labels = ['Cat', 'Dog']
    
    print("-" * 30)
    print(f"Prediction: {labels[prediction]}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    # Update these paths to match your files
    ONNX_MODEL_PATH = 'models/cat_dog_model.onnx'
    
    # Let's try it with a dog from your validation set
    TEST_IMAGE_PATH = 'test_image_cat.jpg' 
    
    if os.path.exists(TEST_IMAGE_PATH):
        test_prediction(TEST_IMAGE_PATH, ONNX_MODEL_PATH)
    else:
        print(f"File not found: {TEST_IMAGE_PATH}")
        print("Please check the path to your test image.")