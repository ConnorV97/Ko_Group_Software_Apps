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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# ##either load pretrained model or train a new model
#
# model = torch.load("pretrained_model/gdm_model.pth", map_location=device)  # Load your trained GDM model

# train_image_path1 = "gold1.png"
# train_image_path2 = "gold_clean1.png"
#
#
# os.makedirs("pretrained_model", exist_ok=True)
#
# model = train(train_image_path1,
#             train_image_path2,
#             w1=0.5,
#             w2=0.5,
#             epochs=50,
#             batch_size=8,
#             patch_size=128,
#             lr=1e-4,
#             mask_fraction=0.1,
#             isfft=True,
#             device=device)
#
# torch.save(model.state_dict(), "pretrained_model/gdm_model.pth")
#
# noisy_image_path = "gold2.png"  # Path to your noisy test image
# img = Image.open(noisy_image_path).convert("RGB")
# den = denoise_image(model, img, patch_size=256, stride=128, device=device) # stride can be adjusted based on the desired overlap between patches. here we use 50% overlap (256 pixels) for a patch size of 512.
#
# den.save("denoised_output.png")  # Save the denoised image