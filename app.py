import tkinter as tk
import warnings
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFile
import torch
from torchvision import transforms
from src.model_builder import CatDogClassifier

# Ignore the specific Pillow warning about truncated files
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

# Tell Pillow it's okay to load these files
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CatDogApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Cat vs Dog Classifier")
        self.window.geometry("500x600")

        # 1. Load Model
        self.device = torch.device("cpu")
        self.model = CatDogClassifier()
        self.model.load_state_dict(torch.load('models/cat_dog_cnn.pth', map_location=self.device))
        self.model.eval()

        # 2. UI Elements
        self.label = tk.Label(window, text="Upload an image to classify", font=("Arial", 14))
        self.label.pack(pady=20)

        self.canvas = tk.Canvas(window, width=300, height=300, bg="gray")
        self.canvas.pack()

        self.btn = tk.Button(window, text="Select Image", command=self.upload_image, font=("Arial", 12))
        self.btn.pack(pady=20)

        self.result_label = tk.Label(window, text="", font=("Arial", 16, "bold"))
        self.result_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        try:
            # Display Image
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            self.img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(150, 150, image=self.img_tk)

            # Predict
            self.predict(file_path)
            
        except Exception as e:
            self.result_label.config(text=f"Error: Could not read image.")
            print(f"File error: {e}")

    def predict(self, path):
        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(prob, 1)

        classes = ['Cat', 'Dog']
        result = classes[predicted.item()]
        score = confidence.item() * 100

        self.result_label.config(text=f"Prediction: {result} ({score:.2f}%)")

if __name__ == "__main__":
    root = tk.Tk()
    app = CatDogApp(root)
    root.mainloop()