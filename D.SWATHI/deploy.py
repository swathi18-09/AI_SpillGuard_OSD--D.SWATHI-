# deploy.py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import gradio as gr
from torchvision import models

# ===============================
# Model (same as in app.py)
# ===============================
class SimpleUNet(nn.Module):
    def __init__(self, n_classes=1, pretrained=True):
        super().__init__()
        self.base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.base_layers = list(self.base_model.children())
        self.enc1 = nn.Sequential(*self.base_layers[:3])
        self.enc2 = nn.Sequential(*self.base_layers[3:5])
        self.enc3 = self.base_layers[5]
        self.enc4 = self.base_layers[6]
        self.enc5 = self.base_layers[7]
        self.center = nn.Sequential(nn.Conv2d(512,512,3,padding=1), nn.ReLU(inplace=True),
                                    nn.Conv2d(512,512,3,padding=1), nn.ReLU(inplace=True))
        self.dec5 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.dec4 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.dec3 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.dec2 = nn.ConvTranspose2d(64,64,2,stride=2)
        self.dec1 = nn.Conv2d(64,n_classes,1)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(e1); e3=self.enc3(e2)
        e4=self.enc4(e3); e5=self.enc5(e4)
        c=self.center(e5)
        d5=self.dec5(c)+e4; d4=self.dec4(d5)+e3
        d3=self.dec3(d4)+e2; d2=self.dec2(d3)+e1
        out=self.dec1(d2)
        out=self.final_upsample(out)
        out=torch.nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

# ===============================
# Load model
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleUNet(n_classes=1).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ===============================
# Preprocess function
# ===============================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def preprocess(img):
    img = np.array(img)[:,:,::-1] # RGB->BGR
    img = cv2.resize(img,(512,512))
    img = img.astype(np.float32)/255.0
    img = (img - IMAGENET_MEAN)/IMAGENET_STD
    img = np.transpose(img,(2,0,1))
    img = np.expand_dims(img,0)
    return torch.tensor(img,dtype=torch.float32)

# ===============================
# Prediction function
# ===============================
def predict(image):
    x = preprocess(image).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(x)).cpu().numpy()[0,0]
        mask = (pred>0.5).astype(np.uint8)*255
    oil_present = "Yes" if mask.sum()>0 else "No"
    mask_img = Image.fromarray(mask)
    return oil_present, mask_img

# ===============================
# Gradio interface
# ===============================
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Oil Present?"), gr.Image(label="Predicted Mask")],
    title="Oil Spill Detection",
    description="Upload an image of water/ocean. The model predicts if oil is present and shows the mask."
)
iface.launch()
