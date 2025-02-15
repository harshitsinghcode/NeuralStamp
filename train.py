import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.data_loader import WatermarkDataset  # Fixed path for the dataset class
from models.watermarking_model import WatermarkingModel  # Fixed model import

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224, as expected by ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet-style normalization
])

# Load Train Dataset
train_folder = "VOCdevkit/VOC2012/train"
watermarked_folder = "VOCdevkit/VOC2012/train_watermarked"
train_dataset = WatermarkDataset(train_folder, watermarked_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize Model
model = WatermarkingModel().to(device)
criterion = nn.BCEWithLogitsLoss()  # For binary classification (watermarked or not)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimizer step

        total_loss += loss.item()

        # Compute accuracy
        preds = torch.round(torch.sigmoid(outputs)).squeeze()  # Sigmoid output and rounding for binary classification
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)

        # Print progress every 10 iterations
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(train_loader)
    accuracy = (correct_preds / total_preds) * 100
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Save Model
torch.save(model.state_dict(), "watermark_detector.pth")
print("Model saved successfully.")
