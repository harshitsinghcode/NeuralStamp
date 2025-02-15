import os
import numpy as np
import cv2
import pywt
from scipy.fftpack import dct, idct
from pytesseract import image_to_string  # OCR for text extraction

# Define Paths
watermarked_folder = "VOCdevkit/VOC2012/train_watermarked"  # Folder with watermarked images
original_folder = "VOCdevkit/VOC2012/train"  # Folder with original images
output_folder = "VOCdevkit/VOC2012/extracted_watermarks"  # Folder to save extracted watermarks
output_txt_folder = "VOCdevkit/VOC2012/extracted_watermark_texts"  # Folder to save extracted text

# Create output directories if not exists
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_txt_folder, exist_ok=True)

# DWT Functions
def apply_dwt(image):
    return pywt.dwt2(image, "haar")

def apply_idwt(coeffs):
    return pywt.idwt2(coeffs, "haar")

# DCT Functions
def apply_dct(sub_band):
    return dct(dct(sub_band.T, norm="ortho").T, norm="ortho")

def apply_idct(dct_coeff):
    return idct(idct(dct_coeff.T, norm="ortho").T, norm="ortho")

# Extract Watermark from Each Color Channel
def extract_watermark_color(image, original, alpha=0.05):
    extracted_channels = []
    
    for i in range(3):  # Process R, G, B separately
        orig_channel = original[:, :, i]
        wm_channel = image[:, :, i]

        # Apply DWT
        LL_orig, (LH_orig, HL_orig, HH_orig) = apply_dwt(orig_channel)
        LL_wm, (LH_wm, HL_wm, HH_wm) = apply_dwt(wm_channel)

        # Apply DCT
        dct_orig = apply_dct(LH_orig)
        dct_wm = apply_dct(LH_wm)

        # Extract watermark by subtracting DCT coefficients and scaling by alpha
        extracted_wm = (dct_wm - dct_orig) / alpha
        extracted_channels.append(np.clip(extracted_wm, 0, 1) * 255)

    # Merge R, G, B extracted watermarks
    return cv2.merge(extracted_channels)

# Post-process the extracted watermark for better clarity
def post_process_watermark(extracted_wm):
    # Convert to grayscale (if it's a color image) to improve extraction quality
    if len(extracted_wm.shape) == 3:
        extracted_wm = cv2.cvtColor(extracted_wm, cv2.COLOR_BGR2GRAY)

    # Normalize and apply thresholding for better visibility
    _, thresholded = cv2.threshold(extracted_wm, 10, 255, cv2.THRESH_BINARY)

    # Apply some morphological operations (e.g., dilation, erosion) if needed
    kernel = np.ones((3, 3), np.uint8)
    processed_wm = cv2.dilate(thresholded, kernel, iterations=1)
    
    return processed_wm

# Perform OCR on Extracted Watermark to Extract Text
def extract_text_from_watermark(extracted_wm):
    # Ensure the image is in 8-bit (grayscale or RGB) mode before OCR
    if extracted_wm.dtype != np.uint8:
        extracted_wm = np.uint8(extracted_wm)

    # Perform OCR (text extraction)
    extracted_text = image_to_string(extracted_wm)
    return extracted_text.strip()

# Process All Watermarked Images
for filename in os.listdir(watermarked_folder):
    if filename.endswith(".jpg"):
        watermarked_path = os.path.join(watermarked_folder, filename)
        original_path = os.path.join(original_folder, filename)
        output_img_path = os.path.join(output_folder, filename)
        output_txt_path = os.path.join(output_txt_folder, filename.replace(".jpg", ".txt"))

        # Read Images in Color (RGB)
        watermarked = cv2.imread(watermarked_path)
        original = cv2.imread(original_path)

        if watermarked is None or original is None:
            print(f"❌ Could not read image: {filename}")
            continue

        # Resize and Normalize (to [0, 1] range)
        watermarked = cv2.resize(watermarked, (512, 512)).astype(np.float32) / 255.0
        original = cv2.resize(original, (512, 512)).astype(np.float32) / 255.0

        # Extract Watermark
        extracted_wm = extract_watermark_color(watermarked, original)
        
        # Post-process the extracted watermark for better clarity
        processed_wm = post_process_watermark(extracted_wm)

        # Save extracted watermark image
        cv2.imwrite(output_img_path, processed_wm.astype(np.uint8))
        print(f"✅ Extracted watermark image saved: {output_img_path}")

        # Perform OCR on the processed watermark image to extract text
        extracted_text = extract_text_from_watermark(processed_wm)

        # Save extracted text to a notepad file
        with open(output_txt_path, "w") as txt_file:
            txt_file.write(extracted_text)

        print(f"✅ Extracted watermark text saved: {output_txt_path}")

print("Watermark extraction and text retrieval completed for all test images.")
