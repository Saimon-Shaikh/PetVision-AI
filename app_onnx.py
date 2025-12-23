import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import onnxruntime as ort
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

# --- HELPER FOR EXE PATHS ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- UI COLORS ---
BG_COLOR = "#212121"
CARD_COLOR = "#2d2d2d"
TEXT_COLOR = "#FFFFFF"
ACCENT_COLOR = "#4CAF50"

class CatDogAppONNX:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Pet Classifier Pro")
        self.root.geometry("500x700")
        self.root.configure(bg=BG_COLOR)

        icon_path = resource_path("favicon.ico")
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)

        # UPDATED: Use resource_path so the EXE can find the model
        model_path = resource_path(os.path.join('models', 'cat_dog_model.onnx'))
        
        try:
            self.session = ort.InferenceSession(model_path)
        except Exception as e:
            print(f"Model Load Error: {e}")
            self.session = None

        # [ ... Rest of your UI code remains exactly the same as before ... ]
        self.header = tk.Label(root, text="Pet Classifier", font=("Segoe UI", 24, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR)
        self.header.pack(pady=(30, 10))
        
        self.image_frame = tk.Frame(root, bg=CARD_COLOR, bd=2, relief="flat")
        self.image_frame.pack(pady=10, padx=50, fill="both", expand=True)

        self.canvas = tk.Canvas(self.image_frame, width=300, height=300, bg=CARD_COLOR, highlightthickness=0)
        self.canvas.pack(pady=20)

        self.result_label = tk.Label(root, text="Ready for Upload", font=("Segoe UI", 16, "bold"), bg=BG_COLOR, fg=TEXT_COLOR)
        self.result_label.pack(pady=10)

        self.conf_label = tk.Label(root, text="Select an image to analyze", font=("Segoe UI", 11), bg=BG_COLOR, fg="#aaaaaa")
        self.conf_label.pack(pady=(0, 20))

        self.upload_btn = tk.Button(root, text="UPLOAD IMAGE", command=self.upload_image, font=("Segoe UI", 12, "bold"),bg=ACCENT_COLOR, fg="white", relief="flat", cursor="hand2", padx=40, pady=10)
        self.upload_btn.pack(pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path: return
        img = Image.open(file_path).convert('RGB')
        display_img = img.copy()
        display_img.thumbnail((300, 300))
        self.img_tk = ImageTk.PhotoImage(display_img)
        self.canvas.delete("all")
        self.canvas.create_image(150, 150, image=self.img_tk)
        if self.session: self.predict(img)

    def predict(self, img):
        img_input = img.resize((128, 128), resample=Image.BICUBIC)
        img_data = np.array(img_input).astype('float32') / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype='float32')
        std = np.array([0.229, 0.224, 0.225], dtype='float32')
        img_data = (img_data - mean) / std
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0).astype('float32')
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
        logits = outputs[0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        idx = np.argmax(probs)
        conf = probs[0][idx] * 100
        label = "CAT" if idx == 0 else "DOG"
        res_color = "#4fc3f7" if label == "DOG" else "#ffb74d"
        self.result_label.config(text=f"IT'S A {label}!", fg=res_color)
        self.conf_label.config(text=f"Confidence Score: {conf:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False, False)
    app = CatDogAppONNX(root)
    root.mainloop()