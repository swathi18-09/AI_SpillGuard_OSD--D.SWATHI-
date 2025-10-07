import torch
from torchvision import models
import torch.nn as nn
import gradio as gr

# Mount Drive in Colab before running
from google.colab import drive
drive.mount('/content/drive')

# Use same model class as in app.py
class SimpleUNet(nn.Module):
    # Paste your SimpleUNet definition here
    ...

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = SimpleUNet(n_classes=1).to(device)
model_path = "/content/drive/MyDrive/OilSpill/best_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Prediction function
def predict(img):
    import cv2
    import numpy as np
    from albumentations.pytorch import ToTensorV2
    import albumentations as A
    
    # Transform image same as eval
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    out = transform(image=img)
    input_tensor = out['image'].unsqueeze(0).to(device, dtype=torch.float32)
    
    with torch.no_grad():
        pred = torch.sigmoid(model(input_tensor)).cpu().numpy()[0,0]
        mask = (pred > 0.5).astype(np.uint8)
    
    # If any oil detected
    oil_detected = mask.sum() > 0
    return mask, "Oil detected" if oil_detected else "No oil detected"

# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="numpy"), gr.Label(num_top_classes=1)],
    title="Oil Spill Detection"
)

iface.launch()
