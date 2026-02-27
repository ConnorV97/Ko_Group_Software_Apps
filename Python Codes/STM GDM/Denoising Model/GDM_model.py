import os, random
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Cal fft similarity score between predicted and target patches
def fft_quality(pred, target):
    """
    FFT-based cosine similarity between predicted and target images.
    Returns a score in [0, 1] (higher = better).
    """
    # Compute FFT
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)

    # Shift zero-frequency to center (for interpretability)
    pred_fft = torch.fft.fftshift(pred_fft)
    target_fft = torch.fft.fftshift(target_fft)

    # Magnitude spectra
    pred_mag = torch.abs(pred_fft).flatten()
    target_mag = torch.abs(target_fft).flatten()

    # Cosine similarity in frequency domain
    score = torch.dot(pred_mag, target_mag) / (
        pred_mag.norm() * target_mag.norm() + 1e-8
    )

    return score.item()

# Function to create random mask on the image patches
def create_random_mask(batch, mask_fraction=0.005):
    B, C, H, W = batch.shape
    Npix = H * W
    nm = max(1, int(mask_fraction * Npix))
    masks = torch.zeros_like(batch)
    for b in range(B):
        idxs = torch.randperm(Npix)[:nm]
        ys = idxs // W
        xs = idxs % W
        masks[b, 0, ys, xs] = 1.0
    return masks

# -------------------- Simple U-Net model --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DoubleConv(base*6, base*2)
        self.dec2 = DoubleConv(base*3, base)
        self.final = nn.Conv2d(base, in_ch, kernel_size=1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d3 = self.up(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        out = self.final(d2)
        return out

# Function to generate image patches from single training image
class STMNoiseSingleImageDataset(Dataset): #input is .png image
    def __init__(self, image_path, patch_size=128):
        if not os.path.exists(image_path):
            raise RuntimeError(f"File not found: {image_path}")
        self.img = Image.open(image_path).convert("L")
        self.patch_size = patch_size
        self.len = 1000  # number of random patches per epoch

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        w, h = self.img.size
        if w < self.patch_size or h < self.patch_size:
            self.img = self.img.resize((max(w, self.patch_size), max(h, self.patch_size)))
            w, h = self.img.size
        left = random.randint(0, w - self.patch_size)
        top = random.randint(0, h - self.patch_size)
        patch = self.img.crop((left, top, left + self.patch_size, top + self.patch_size))
        patch = np.array(patch).astype(np.float32) / 255.0
        patch = torch.from_numpy(patch).unsqueeze(0)
        return patch
    
# Function to train the GDM model
def train(train_image_path1, train_image_path2, w1=0.5, w2=0.5, epochs=50, batch_size=8, patch_size=128, lr=1e-4, mask_fraction=0.1,isfft=True,device='cuda'):

    path1 = train_image_path1
    path2 = train_image_path2

    dataset1 = STMNoiseSingleImageDataset(path1, patch_size=patch_size)
    loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    dataset2 = STMNoiseSingleImageDataset(path2, patch_size=patch_size)  ### FIXED
    loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = UNetSmall(in_ch=1, base=32).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)
    mse = nn.MSELoss(reduction='none')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_fft_score = 0.0

        # zip ensures parallel iteration (stops at shortest loader)
        pbar = tqdm(zip(loader1, loader2), total=min(len(loader1), len(loader2)), desc=f"Epoch {epoch+1}/{epochs}")
        #j = 0
        for patches1, patches2 in pbar:
            #j = j+1
            # ---------------- dataset1 ----------------
            patches1 = patches1.to(device)
            masks1 = create_random_mask(patches1, mask_fraction=mask_fraction).to(device)
            inp1 = patches1.clone()
            inp1 = inp1 * (1.0 - masks1) + 0.5 * masks1
            #print(inp1.shape)
            out1 = model(inp1)
            loss_map1 = mse(out1, patches1).mean(dim=1, keepdim=True)
            pixel_loss1 = (loss_map1 * masks1).sum() / (masks1.sum() + 1e-8)
            fft_score1 = fft_quality(out1, patches1)

            # ---------------- dataset2 ----------------
            patches2 = patches2.to(device)
            masks2 = create_random_mask(patches2, mask_fraction=mask_fraction).to(device)
            inp2 = patches2.clone()
            inp2 = inp2 * (1.0 - masks2) + 0.5 * masks2
            out2 = model(inp2)
            #print(inp1.shape)
            loss_map2 = mse(out2, patches2).mean(dim=1, keepdim=True)
            pixel_loss2 = (loss_map2 * masks2).sum() / (masks2.sum() + 1e-8)
            fft_score2 = fft_quality(out2, patches2)

            # ---------------- combine ----------------
            w_pixel_loss = w1 * pixel_loss1 + w2 * pixel_loss2
            w_fft_score = w1 * fft_score1 + w2 * fft_score2

            if isfft == True:
              loss = w_pixel_loss / (1 + w_fft_score)
            else:
              loss = w_pixel_loss
              w_fft_score = 0
            #loss = w_pixel_loss / (1 + fft_score2)
            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_fft_score += w_fft_score
            #running_fft_score += fft_score2

            pbar.set_postfix({
                "loss": f"{running_loss:.4f}",
                "fft_score": f"{running_fft_score / (pbar.n+1):.4f}"
            })
        #print(j)
        scheduler.step()

    return model

if __name__ == "__main__":
    # Example paths
    image_path1 = r"C:\Users\conno.DESKTOP-98EBONR\Downloads\20251217_Au(111)_4K_Auto_Mag-20260219T232221Z-1-001\20251217_Au(111)_4K_Auto_Mag\flatten\20251217_172037_scan001_Au(111)_4k_STM_AUTO_(Both)_0.5T_Au(111)_0064_flat.png"

    image_path2 = r"C:\Users\conno.DESKTOP-98EBONR\Downloads\20251217_Au(111)_4K_Auto_Mag-20260219T232221Z-1-001\20251217_Au(111)_4K_Auto_Mag\flatten\20251217_202632_scan025_Au(111)_4k_STM_AUTO_(Both)_0.5T_Au(111)_0094_flat.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train(image_path1, image_path2,
                  w1=0.5, w2=0.5,
                  epochs=50,
                  batch_size=8,
                  patch_size=128,
                  lr=1e-4,
                  mask_fraction=0.1,
                  isfft=True,
                  device=torch.device(device))
    os.makedirs("pretrained_model", exist_ok=True)
    torch.save(model.state_dict(), "pretrained_model/gdm_model.pth")




