import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.data_loader import WatermarkDataset
from models.watermarking_model import WatermarkingModel

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Train Dataset
train_folder = "VOCdevkit/VOC2012/train"
watermarked_folder = "VOCdevkit/VOC2012/train_watermarked"
train_dataset = WatermarkDataset(train_folder, watermarked_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize Model and load previous weights
model = WatermarkingModel().to(device)
model.load_state_dict(torch.load("watermark_detector_epoch_16.pth"))
print("Previous model weights loaded successfully")

# Initialize loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduced learning rate for fine-tuning

# Continue Training
start_epoch = 16  # Since you've already trained for 10 epochs
num_additional_epochs = 10
total_epochs = start_epoch + num_additional_epochs

print(f"Continuing training from epoch {start_epoch + 1}...")

for epoch in range(start_epoch, total_epochs):
    model.train()
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), labels.float())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Compute accuracy
        preds = torch.round(torch.sigmoid(outputs)).squeeze()
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)
        
        # Print progress every 10 iterations
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{total_epochs}], Step [{batch_idx}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")
    
    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(train_loader)
    accuracy = (correct_preds / total_preds) * 100
    
    print(f"Epoch [{epoch+1}/{total_epochs}], Average Loss: {avg_loss:.4f}, "
          f"Accuracy: {accuracy:.2f}%")
    
    # Save model after each epoch
    torch.save(model.state_dict(), f"watermark_detector_epoch_{epoch+1}.pth")
    print(f"Model saved for epoch {epoch+1}")

# Save final model
torch.save(model.state_dict(), "watermark_detector_final.pth")
print("Final model saved successfully.")