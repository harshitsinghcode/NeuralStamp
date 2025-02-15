import torch
import torch.nn as nn
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torchvision import transforms
from models.watermarking_model import WatermarkingModel  # Import your trained model

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained Model
model = WatermarkingModel().to(device)
model.load_state_dict(torch.load("watermark_detector_final.pth", map_location=device))
model.eval()  # Set model to evaluation mode

# Define Transformations for Model Input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def compute_metrics(original, manipulated):
    """Compute PSNR and SSIM between original and manipulated images"""
    try:
        original_uint8 = (original * 255).astype(np.uint8)  # Convert to uint8
        manipulated_uint8 = (manipulated * 255).astype(np.uint8)

        psnr_value = psnr(original_uint8, manipulated_uint8, data_range=255)
        ssim_value = ssim(original_uint8, manipulated_uint8, 
                          win_size=7, 
                          channel_axis=2,
                          data_range=255)
        return psnr_value, ssim_value
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None, None

def process_image(image_path):
    try:
        # Read and preprocess original image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original = cv2.resize(original, (512, 512))
        original = original.astype(np.float32) / 255.0  # Normalize

        results = {}
        transformed_images = {}

        # 1. Gaussian Noise (Reduced noise strength)
        noise = np.random.normal(0, 0.005, original.shape)  # Decreased variance
        noisy = np.clip(original + noise, 0, 1)
        results['Gaussian Noise'] = compute_metrics(original, noisy)
        transformed_images['Gaussian Noise'] = noisy

        # 2. Blur (Softer blur effect)
        blurred = cv2.GaussianBlur(original, (3, 3), 0.8)  # Smaller kernel and sigma
        results['Blur'] = compute_metrics(original, blurred)
        transformed_images['Blur'] = blurred

        # 3. Compression (Higher JPEG quality)
        temp_path = "temp_compressed.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), 
                    [cv2.IMWRITE_JPEG_QUALITY, 85])  # Higher quality
        compressed = cv2.imread(temp_path)
        compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
        compressed = compressed.astype(np.float32) / 255.0
        results['Compression'] = compute_metrics(original, compressed)
        transformed_images['Compression'] = compressed

        # 4. Cropping (Smaller crop size)
        h, w = original.shape[:2]
        crop_size = int(min(h, w) * 0.9)  # Crop less aggressively
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        cropped = original[start_h:start_h+crop_size, start_w:start_w+crop_size]
        cropped = cv2.resize(cropped, (512, 512))
        results['Cropping'] = compute_metrics(original, cropped)
        transformed_images['Cropping'] = cropped

        return results, transformed_images, original
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None

def detect_watermark(image):
    """Run image through the trained model to check for watermark presence"""
    try:
        # Convert NumPy image to Tensor for model input
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Forward pass through model
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.sigmoid(output).item()
        
        return prediction  # Probability of being watermarked
    except Exception as e:
        # print(f"Error in watermark detection: {e}")
        return None

if __name__ == "__main__":
    image_path = "VOCdevkit/VOC2012/test/2007_000061.jpg"
    
    print(f"Processing image: {image_path}")
    print("-" * 50)
    
    results, transformed_images, original_image = process_image(image_path)

    if results:
        print("\nResults after attacks:")
        for manipulation, (psnr_val, ssim_val) in results.items():
            print(f"\n{manipulation}:")
            if psnr_val is not None:
                print(f"PSNR: {psnr_val:.4f} dB")
            if ssim_val is not None:
                print(f"SSIM: {ssim_val:.4f}")
        
        # Watermark detection for original and manipulated images
        # print("\nWatermark Detection Results:")
        for manipulation, img in transformed_images.items():
            watermark_prob = detect_watermark(img)
            if watermark_prob is not None:
                print(f"{manipulation}: Probability = {watermark_prob:.4f}")

        # Also check watermark for the original image
        watermark_prob_original = detect_watermark(original_image)
        if watermark_prob_original is not None:
            print(f"\nOriginal Image: Probability = {watermark_prob_original:.4f}")
