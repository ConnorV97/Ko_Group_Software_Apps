import torch
import numpy as np
from PIL import Image
from GDM_model import train
import os


# Function to test the GDM model- We call this once we train the GDM and want to denoise a test noisy experimental image. The code returns the denoised image
def denoise_image(model, pil_img, patch_size=128, stride=64, device=torch.device("cpu")):
    model.eval()
    img = pil_img.convert("L")
    arr = np.array(img).astype(np.float32) / 255.0
    H, W = arr.shape
    pad_h = (patch_size - (H % stride)) % stride
    pad_w = (patch_size - (W % stride)) % stride
    padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')
    Ph, Pw = padded.shape
    out_img = np.zeros_like(padded)
    weight = np.zeros_like(padded)
    with torch.no_grad():
        for top in range(0, Ph - patch_size + 1, stride):
            for left in range(0, Pw - patch_size + 1, stride):
                patch = padded[top:top+patch_size, left:left+patch_size]
                x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                pred = model(x).cpu().numpy()[0,0]
                out_img[top:top+patch_size, left:left+patch_size] += pred
                weight[top:top+patch_size, left:left+patch_size] += 1.0
    out_img = out_img / (weight + 1e-8)
    out_img = out_img[:H, :W]
    out_img = (out_img * 255.0).clip(0,255).astype(np.uint8)
    return Image.fromarray(out_img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##either load pretrained model or train a new model

model = torch.load("pretrained_model/gdm_model.pth", map_location=device)  # Load your trained GDM model
