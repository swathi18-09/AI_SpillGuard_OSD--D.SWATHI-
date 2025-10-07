

# STEP 1: Setup & Import Libraries
# =============================
from google.colab import drive
import zipfile, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import Dataset, DataLoader

# Mount Google Drive
drive.mount('/content/drive')

# Path to your uploaded dataset zip in Drive
zip_path = "/content/drive/MyDrive/dataset.zip"

# Unzip dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content/dataset")

# STEP 2: Read label_colors.txt

import os, cv2, numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def load_label_colors_txt(path):
    """File format: R G B name (e.g., '255 0 124 oil')."""
    class_names, label_colors = [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                r, g, b = map(int, parts[:3])
                name = " ".join(parts[3:])
                class_names.append(name)
                label_colors.append((r, g, b))
    return class_names, label_colors

class_names, label_colors = load_label_colors_txt("/content/dataset/label_colors.txt")
print("Classes:", class_names)
print("Colors:", label_colors)  # Expect: background, oil, others, water

# STEP 3: Mask conversion utils

def rgb_to_mask(mask_rgb, label_colors):
    """RGB color mask -> class index mask (H, W)."""
    h, w = mask_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.int64)
    for idx, color in enumerate(label_colors):
        mask[(mask_rgb == color).all(axis=-1)] = idx
    return mask

def mask_to_rgb(mask, label_colors):
    """Class index mask -> RGB color mask for viz."""
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(label_colors):
        rgb[mask == idx] = color
    return rgb


# STEP 4: Transforms (512x512)

def get_transforms(image_size=(512, 512), is_train=True):
    ops = [A.Resize(*image_size)]
    if is_train:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
        ]
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)

train_tf = get_transforms((512, 512), is_train=True)
eval_tf  = get_transforms((512, 512), is_train=False)

# STEP 5: Dataset (pairs by basename; JPG<->PNG safe; binary switch)

class SegmentationDataset(Dataset):
    """
    binary=False -> multiclass (0:background,1:oil,2:others,3:water)
    binary=True  -> Oil vs Non-Oil (1 vs 0)
    """
    def __init__(self, img_dir, mask_dir, label_colors, transform, binary=False):
        self.label_colors = label_colors
        self.transform = transform
        self.binary = binary

        img_files  = [f for f in os.listdir(img_dir) if not f.startswith('.')]
        mask_files = [f for f in os.listdir(mask_dir) if not f.startswith('.')]
        img_map  = {os.path.splitext(f)[0]: os.path.join(img_dir, f) for f in img_files}
        mask_map = {os.path.splitext(f)[0]: os.path.join(mask_dir, f) for f in mask_files}

        keys = sorted(set(img_map.keys()) & set(mask_map.keys()))
        self.pairs = [(img_map[k], mask_map[k]) for k in keys]

        missing_imgs = sorted(set(mask_map.keys()) - set(img_map.keys()))
        missing_masks = sorted(set(img_map.keys()) - set(mask_map.keys()))
        if missing_imgs:  print("⚠️ Masks without images (first 10):", missing_imgs[:10])
        if missing_masks: print("⚠️ Images without masks (first 10):", missing_masks[:10])

        # cache oil class index (assumes class_names order used earlier)
        # Find 'oil' idx based on color list order you loaded
        self.oil_idx = 1  # from your file order: background=0, oil=1, others=2, water=3

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]


        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        mask_rgb = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

        mask = rgb_to_mask(mask_rgb, self.label_colors)


        if self.binary:
            mask = (mask == self.oil_idx).astype(np.uint8)

        out = self.transform(image=img, mask=mask)
        img_t, mask_t = out["image"], out["mask"]
        return img_t, mask_t

# STEP 6: DataLoaders

train_dataset = SegmentationDataset(
    img_dir="/content/dataset/train/images",
    mask_dir="/content/dataset/train/masks",
    label_colors=label_colors,
    transform=train_tf,
    binary=True,    # <<<<<<<<<<<<<<<<<<  Oil vs Non-Oil
)

val_dataset = SegmentationDataset(
    img_dir="/content/dataset/val/images",
    mask_dir="/content/dataset/val/masks",
    label_colors=label_colors,
    transform=eval_tf,
    binary=True,
)

test_dataset = SegmentationDataset(
    img_dir="/content/dataset/test/images",
    mask_dir="/content/dataset/test/masks",
    label_colors=label_colors,
    transform=eval_tf,
    binary=True,
)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)




def denorm(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Tensor CxHxW -> HxWxC in [0,1] for display."""
    img = img_t.permute(1,2,0).cpu().numpy()
    img = (img * np.array(std)) + np.array(mean)
    return np.clip(img, 0, 1)

def show_batch(loader, label_colors, binary=True, n=4):
    imgs, masks = next(iter(loader))
    n = min(n, imgs.shape[0])
    plt.figure(figsize=(4*n, 8))
    for i in range(n):
        img = denorm(imgs[i])
        mask = masks[i].cpu().numpy()

        plt.subplot(2, n, i+1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(2, n, n+i+1)
        if binary:
            plt.imshow(mask, cmap="gray")
            plt.title("Mask (Oil=1)")
        else:
            plt.imshow(mask_to_rgb(mask, label_colors))
            plt.title("Mask (multiclass)")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_batch(train_loader, label_colors, binary=True, n=4)





import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

fig = plt.figure(figsize=(4, 3), facecolor='w')
plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)
plt.title("Sample Visualization", fontsize=10)

data = io.BytesIO()
plt.savefig(data)
image = F"data:image/png;base64,{base64.b64encode(data.getvalue()).decode()}"
alt = "Sample Visualization"
display.display(display.Markdown(F"""![{alt}]({image})"""))
plt.close(fig)
# STEP 7: Model, Loss, Optimizer
# =============================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Example: Simple UNet-like model using torchvision resnet backbone
class SimpleUNet(nn.Module):
    def __init__(self, n_classes=1, pretrained=True):
        super().__init__()
        self.base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.base_layers = list(self.base_model.children())

        self.enc1 = nn.Sequential(*self.base_layers[:3])   # 64
        self.enc2 = nn.Sequential(*self.base_layers[3:5])  # 64
        self.enc3 = self.base_layers[5]                    # 128
        self.enc4 = self.base_layers[6]                    # 256
        self.enc5 = self.base_layers[7]                    # 512

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.dec5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        c = self.center(e5)

        d5 = self.dec5(c) + e4
        d4 = self.dec4(d5) + e3
        d3 = self.dec3(d4) + e2
        d2 = self.dec2(d3) + e1
        out = self.dec1(d2)
        return out

model = SimpleUNet(n_classes=1).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# STEP 8: Training & Validation Loop
# =================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32).unsqueeze(1)  # (B,1,H,W)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32).unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

# STEP 9: Training Loop
# ====================

num_epochs = 10
best_val_loss = float("inf")

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = eval_model(model, val_loader, criterion, device)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "/content/best_model.pth")
        print("✅ Saved Best Model!")

# STEP 10: Test Inference & Visualization
# =======================================

model.load_state_dict(torch.load("/content/best_model.pth"))
model.eval()

def predict_and_show(model, loader, n=4):
    imgs, masks = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        preds = torch.sigmoid(model(imgs)).cpu().numpy()
    
    plt.figure(figsize=(4*n, 8))
    for i in range(min(n, imgs.shape[0])):
        img = denorm(imgs[i])
        mask = masks[i].cpu().numpy()
        pred = (preds[i,0] > 0.5).astype(np.uint8)
        
        plt.subplot(3, n, i+1)
        plt.imshow(img); plt.axis("off"); plt.title("Image")
        plt.subplot(3, n, n+i+1)
        plt.imshow(mask, cmap="gray"); plt.axis("off"); plt.title("Mask")
        plt.subplot(3, n, 2*n+i+1)
        plt.imshow(pred, cmap="gray"); plt.axis("off"); plt.title("Pred")
    plt.tight_layout()
    plt.show()

predict_and_show(model, test_loader, n=4)

# STEP 11: Metrics for Binary Segmentation
# =======================================

def dice_score(pred, target, eps=1e-6):
    """pred & target are binary numpy arrays (0/1)"""
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = (pred * target).sum()
    return (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-6):
    """pred & target are binary numpy arrays (0/1)"""
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

# STEP 12: Updated Evaluation with Metrics
# ========================================

def eval_model_metrics(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_total = 0.0
    iou_total = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32).unsqueeze(1)
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * imgs.size(0)
            
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            for p, t in zip(preds, masks_np):
                dice_total += dice_score(p[0], t[0])
                iou_total  += iou_score(p[0], t[0])
    
    n = len(loader.dataset)
    avg_loss = running_loss / n
    avg_dice = dice_total / n
    avg_iou  = iou_total / n
    return avg_loss, avg_dice, avg_iou

# STEP 13: Training Loop with Metrics
# ==================================

num_epochs = 10
best_val_dice = 0.0  # Track best Dice score

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_dice, val_iou = eval_model_metrics(model, val_loader, criterion, device)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
    
    # Save best model based on Dice
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "/content/best_model.pth")
        print("✅ Saved Best Model based on Dice score!")

# STEP 14: Test Set Metrics
# ========================

model.load_state_dict(torch.load("/content/best_model.pth"))
test_loss, test_dice, test_iou = eval_model_metrics(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f} | Test IoU: {test_iou:.4f}")
# STEP 15: Visualization with Dice & IoU per image
# =================================================

def predict_and_show_metrics(model, loader, n=4):
    model.eval()
    imgs, masks = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        preds = torch.sigmoid(model(imgs)).cpu().numpy()
    
    plt.figure(figsize=(4*n, 10))
    for i in range(min(n, imgs.shape[0])):
        img = denorm(imgs[i])
        mask = masks[i].cpu().numpy()
        pred = (preds[i,0] > 0.5).astype(np.uint8)
        
        # Compute metrics per image
        dice = dice_score(pred, mask)
        iou  = iou_score(pred, mask)
        
        plt.subplot(3, n, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Image")
        
        plt.subplot(3, n, n+i+1)
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        plt.title("Mask")
        
        plt.subplot(3, n, 2*n+i+1)
        plt.imshow(pred, cmap="gray")
        plt.axis("off")
        plt.title(f"Pred\nDice={dice:.3f}, IoU={iou:.3f}")
    
    plt.tight_layout()
    plt.show()


predict_and_show_metrics(model, test_loader, n=4)
