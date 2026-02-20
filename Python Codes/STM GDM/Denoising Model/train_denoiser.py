import torch
import os
from GDM_model import train


image_path1 = r"C:\Users\conno.DESKTOP-98EBONR\Downloads\20251217_Au(111)_4K_Auto_Mag-20260219T232221Z-1-001\20251217_Au(111)_4K_Auto_Mag\flatten\20251217_172037_scan001_Au(111)_4k_STM_AUTO_(Both)_0.5T_Au(111)_0064_flat.png"
image_path2 = r"C:\Users\conno.DESKTOP-98EBONR\Downloads\20251217_Au(111)_4K_Auto_Mag-20260219T232221Z-1-001\20251217_Au(111)_4K_Auto_Mag\flatten\20251217_202632_scan025_Au(111)_4k_STM_AUTO_(Both)_0.5T_Au(111)_0094_flat.png"

def train_model(image_path1, image_path2):
    train_image_path1 = image_path1
    train_image_path2 = image_path2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    os.makedirs("pretrained_model", exist_ok=True)

    model = train(train_image_path1,
                train_image_path2,
                w1=0.5, 
                w2=0.5, 
                epochs=50, 
                batch_size=8, 
                patch_size=128, 
                lr=1e-4, 
                mask_fraction=0.1, 
                isfft=True,
                device=device)

    torch.save(model.state_dict(), "pretrained_model/model.pth")
    return model