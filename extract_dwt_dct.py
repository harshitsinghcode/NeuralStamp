import os
import numpy as np
import cv2
import pywt
from scipy.fftpack import dct, idct
from pytesseract import image_to_string  # OCR for text extraction

# Define Paths
watermarked_folder = "VOCdevkit/VOC2012/train_watermarked"
original_folder = "VOCdevkit/VOC2012/train"
output_folder = "VOCdevkit/VOC2012/extracted_watermarks"
output_txt_folder = "VOCdevkit/VOC2012/extracted_watermark_texts"

# Create output directories if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_txt_folder, exist_ok=True)

# DWT and DCT Functions
def apply_dwt(image):
    return pywt.dwt2(image, "haar")

def apply_idwt(coeffs):
    return pywt.idwt2(coeffs, "haar")

def apply_dct(sub_band):
    return dct(dct(sub_band.T, norm="ortho").T, norm="ortho")

def apply_idct(dct_coeff):
    return idct(idct(dct_coeff.T, norm="ortho").T, norm="ortho")

# Extract Watermark from Color Image
def extract_watermark(image, original, alpha=0.05):
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

        # Normalize and scale pixel values
        extracted_wm = np.clip(extracted_wm, 0, 1) * 255
        extracted_channels.append(extracted_wm)

    # Convert back to a grayscale image
    extracted_wm = np.mean(extracted_channels, axis=0)
    return extracted_wm.astype(np.uint8)

# Improve OCR Readability
def enhance_watermark(extracted_wm):
    # Convert to grayscale if not already
    if len(extracted_wm.shape) == 3:
        extracted_wm = cv2.cvtColor(extracted_wm, cv2.COLOR_BGR2GRAY)

    # Apply Adaptive Histogram Equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(extracted_wm)

    # Apply Adaptive Thresholding
    enhanced = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    
    return enhanced

# Perform OCR on Extracted Watermark
def extract_text_from_watermark(extracted_wm):
    # Ensure 8-bit grayscale format
    if extracted_wm.dtype != np.uint8:
        extracted_wm = np.uint8(extracted_wm)

    # Run OCR
    extracted_text = image_to_string(extracted_wm, config='--psm 6')  # psm 6 = Assume a single uniform text block
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

        # Resize and Normalize
        watermarked = cv2.resize(watermarked, (512, 512)).astype(np.float32) / 255.0
        original = cv2.resize(original, (512, 512)).astype(np.float32) / 255.0

        # Extract Watermark
        extracted_wm = extract_watermark(watermarked, original)

        # Enhance Watermark for Better OCR Performance
        enhanced_wm = enhance_watermark(extracted_wm)

        # Save extracted watermark image
        cv2.imwrite(output_img_path, enhanced_wm)
        print(f"✅ Extracted watermark image saved: {output_img_path}")

        # Perform OCR on the processed watermark image to extract text
        extracted_text = extract_text_from_watermark(enhanced_wm)

        # Save extracted text
        with open(output_txt_path, "w") as txt_file:
            txt_file.write(extracted_text)

        print(f"✅ Extracted watermark text saved: {output_txt_path}")

print("✅ Watermark extraction and text retrieval completed for all test images.")
