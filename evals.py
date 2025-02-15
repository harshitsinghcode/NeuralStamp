import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.watermarking_model import WatermarkingModel  # Import your custom watermarking model
from utils.data_loader import WatermarkDataset  # Import your custom dataset loader

# ---------------------
# Set up the image transformations for preprocessing
# ---------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (common input size for many models)
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # Normalize images using mean and standard deviation values (commonly used for pre-trained networks)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------
# Define the paths to your data folders
# ---------------------
clean_folder = "VOCdevkit/VOC2012/train"            # Folder with original (clean) images
watermarked_folder = "VOCdevkit/VOC2012/train_watermarked"  # Folder with watermarked images

# ---------------------
# Create the dataset and dataloader for evaluation
# ---------------------
# The WatermarkDataset should load both clean and watermarked images along with their labels.
test_dataset = WatermarkDataset(clean_folder, watermarked_folder, transform=transform)
# DataLoader will handle batching (here, 32 images per batch) and shuffling (set to False for evaluation)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------------
# Load the watermarking model and its saved weights
# ---------------------
model = WatermarkingModel()
# Load the model's state dictionary (ensure that the path to 'watermark_detector_final.pth' is correct)
model.load_state_dict(torch.load('watermark_detector_final.pth'))
model.eval()  # Set the model to evaluation mode to disable dropout and batch normalization updates

# ---------------------
# Set the device to GPU if available, otherwise default to CPU
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to the specified device

# ---------------------
# Begin the evaluation loop over the test dataset
# ---------------------
correct = 0  # Counter for correct predictions
total = 0    # Counter for total number of samples

# Disable gradient calculation for efficiency during evaluation
with torch.no_grad():
    # Iterate over each batch in the test loader
    for images, labels in test_loader:
        images = images.to(device)  # Move images to the device
        labels = labels.to(device)  # Move labels to the device
        
        # Forward pass: get model outputs for the batch
        outputs = model(images)
        
        # Apply a threshold to convert outputs to binary predictions (1: watermarked, 0: non-watermarked)
        predicted = (outputs > 0.5).float()
        
        # Update counters: total number of samples and number of correct predictions
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()

# Calculate the overall accuracy in percentage
accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
