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
#
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

# def plane(coords, a, b, c):
#     """
#     Defines a plane function: ax + by + c.
#     """
#     x, y = coords
#     return a * x + b * y + c
#
#
# def poly(coords, a, b, c, d, e, f):
#     """
#     Defines a polynomial surface function: ax^2 + by^2 + cxy + dx + ey + f.
#     """
#     x, y = coords
#     return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f
#
#
# def FitPlane(img, mask=None):
#     """
#     Fits a plane surface to the image data, optionally applying a mask to focus on certain regions.
#     Returns the fitted surface and the Pearson correlation.
#     """
#     x = np.arange(img.shape[1])
#     y = np.arange(img.shape[0])
#     x, y = np.meshgrid(x, y)
#
#     if mask is None:
#         x_flat, y_flat, z_flat = x.flatten(), y.flatten(), img.flatten()
#     else:
#         x_flat, y_flat, z_flat = x[mask == 1].flatten(), y[mask == 1].flatten(), img[mask == 1].flatten()
#
#     p0 = np.zeros(3)
#     params, _ = curve_fit(plane, (x_flat, y_flat), z_flat, p0)
#     plane_fitted = plane((x, y), *params).reshape(img.shape)
#     correlation, _ = pearsonr(z_flat, plane((x_flat, y_flat), *params))
#     return plane_fitted, correlation
#
#
# def FitPoly(img, mask=None):
#     """
#     Fits a polynomial surface to the image data, optionally applying a mask to focus on certain regions.
#     Returns the fitted surface and the Pearson correlation.
#     """
#     x = np.arange(img.shape[1])
#     y = np.arange(img.shape[0])
#     x, y = np.meshgrid(x, y)
#
#     if mask is None:
#         x_flat, y_flat, z_flat = x.flatten(), y.flatten(), img.flatten()
#     else:
#         x_flat, y_flat, z_flat = x[mask == 1].flatten(), y[mask == 1].flatten(), img[mask == 1].flatten()
#
#     p0 = np.zeros(6)
#     params, _ = curve_fit(poly, (x_flat, y_flat), z_flat, p0)
#     poly_fitted = poly((x, y), *params).reshape(img.shape)
#     correlation, _ = pearsonr(z_flat, poly((x_flat, y_flat), *params))
#
#     return poly_fitted, correlation
#
#
# def SubtractGlobalPoly(img, show=False):
#     """
#     Subtracts a polynomial surface from the image and optionally displays the result.
#     """
#     poly_fitted, correlation = FitPoly(img)
#     img_flattened = img - poly_fitted
#
#     if show:
#         fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#         ax[0].imshow(img, cmap='gray')
#         ax[0].set_title('Original Image')
#         ax[1].imshow(img_flattened, cmap='gray')
#         ax[1].set_title('Flattened Image')
#         plt.show()
#
#     return img_flattened, poly_fitted
#
#
# def SubtractGlobalPlane(img, show=False):
#     """
#     Subtracts a fitted plane from the image and optionally displays the result.
#     """
#     plane_fitted, correlation = FitPlane(img)
#     img_flattened = img - plane_fitted
#
#     if show:
#         fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#         ax[0].imshow(img, cmap='gray')
#         ax[0].set_title('Original Image')
#         ax[1].imshow(img_flattened, cmap='gray')
#         ax[1].set_title('Flattened Image')
#         plt.show()
#
#     return img_flattened, plane_fitted
#
#
# def FitOffsetToFlattingImageByDiffAndMask(img, mask=None):
#     """
#     Calculates the offset between adjacent lines in an image using a mask to ignore certain pixels.
#     """
#     offset_img = img.copy()
#     for i in range(1, offset_img.shape[0]):
#         line_below, current_line = offset_img[i - 1, :], offset_img[i, :]
#         if mask is not None:
#             line_below_masked, current_line_masked = line_below[mask[i, :] == 1], current_line[mask[i, :] == 1]
#             offset = np.median(current_line_masked - line_below_masked) if len(line_below_masked) > 0 else 0
#         else:
#             offset = np.median(current_line - line_below)
#         offset_img[i, :] -= offset
#
#     if mask is not None:
#         offset_img[mask == 0] = 0
#
#     return offset_img