import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Define paths
original_folder = "VOCdevkit/VOC2012/test"  # Folder containing original images
watermarked_folder = "VOCdevkit/VOC2012/test_watermarked"  # Folder containing watermarked images

# Initialize variables to calculate average PSNR and SSIM
total_psnr = 0
total_ssim = 0
image_count = 0

# Iterate over all images in the watermarked folder
for filename in os.listdir(watermarked_folder):
    # Construct the full file path for the original and watermarked images
    original_image_path = os.path.join(original_folder, filename)
    watermarked_image_path = os.path.join(watermarked_folder, filename)

    # Read the original and watermarked images in grayscale
    original = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE) / 255.0
    watermarked = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE) / 255.0

    # Check if both images are loaded properly
    if original is None or watermarked is None:
        print(f"Error loading image {filename}, skipping.")
        continue

    # Compute PSNR and SSIM for the current image
    psnr_value = psnr(original, watermarked)
    ssim_value = ssim(original, watermarked, data_range=1.0)

    # Add to the totals for averaging later
    total_psnr += psnr_value
    total_ssim += ssim_value
    image_count += 1

    # Print the metrics for the current image
    print(f"Image: {filename} | PSNR: {psnr_value:.2f} | SSIM: {ssim_value:.4f}")

# Calculate and print the average PSNR and SSIM
if image_count > 0:
    avg_psnr = total_psnr / image_count
    avg_ssim = total_ssim / image_count
    print(f"\nAverage PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
else:
    print("No images processed.")
