import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class WatermarkDataset(Dataset):
    def __init__(self, clean_folder, watermarked_folder, transform=None):
        self.clean_folder = clean_folder
        self.watermarked_folder = watermarked_folder
        self.transform = transform
        
        self.clean_images = [os.path.join(clean_folder, fname) for fname in os.listdir(clean_folder) if fname.endswith(".jpg")]
        self.watermarked_images = [os.path.join(watermarked_folder, fname) for fname in os.listdir(watermarked_folder) if fname.endswith(".jpg")]
        
        # Combine clean and watermarked images
        self.image_paths = self.clean_images + self.watermarked_images
        self.labels = [0] * len(self.clean_images) + [1] * len(self.watermarked_images)  # 0 for clean, 1 for watermarked
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create dataset and dataloader
train_dataset = WatermarkDataset("VOCdevkit/VOC2012/train", "VOCdevkit/VOC2012/train_watermarked", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
