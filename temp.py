import cv2
import numpy as np
import os

# Define paths
original_folder = "VOCdevkit/VOC2012/test"  # Folder containing original images
watermarked_folder = "VOCdevkit/VOC2012/test_watermarked"  # Folder to save watermarked images

# Create the watermarked folder if it doesn't exist
if not os.path.exists(watermarked_folder):
    os.makedirs(watermarked_folder)

# Define the text watermark
def generate_text_watermark(text="SAMSUNG_PRISM 13022025 SHOT ON SAMSUNG PIXEL IPHONE", size=(512, 512)):
    # Create a blank image with the same size as the original image
    watermark = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    # Set font type and size for the watermark text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White color for the text watermark
    thickness = 2

    # Get the size of the text to center it in the image
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (watermark.shape[1] - text_size[0]) // 2
    text_y = (watermark.shape[0] + text_size[1]) // 2

    # Put the watermark text onto the image
    cv2.putText(watermark, text, (text_x, text_y), font, font_scale, color, thickness)

    return watermark

# Define the watermark embedding function (embed text watermark)
def embed_watermark(image, watermark, alpha=0.05):
    # Resize the watermark to the size of the image
    watermark_resized = cv2.resize(watermark, (image.shape[1], image.shape[0]))

    # Add the watermark to the RGB image
    watermarked_image = image + alpha * watermark_resized
    watermarked_image = np.clip(watermarked_image, 0, 255)  # Ensure pixel values are between 0 and 255
    return watermarked_image.astype(np.uint8)

# Process each image in the original folder
for filename in os.listdir(original_folder):
    # Construct the full path for the original image
    original_image_path = os.path.join(original_folder, filename)
    
    # Read the original image in color (RGB)
    original_image = cv2.imread(original_image_path)
    
    if original_image is None:
        print(f"Error loading image {filename}, skipping.")
        continue
    
    # Generate the text watermark
    watermark = generate_text_watermark(text="SAMSUNG_PRISM 13022025 SHOT ON SAMSUNG PIXEL IPHONE", size=(original_image.shape[0], original_image.shape[1]))
    
    # Embed the watermark in the original image
    watermarked_image = embed_watermark(original_image, watermark)
    
    # Save the watermarked image to the new folder
    watermarked_image_path = os.path.join(watermarked_folder, filename)
    cv2.imwrite(watermarked_image_path, watermarked_image)

    print(f"Watermarked image saved: {filename}")

print("Watermarking completed for all test images.")
