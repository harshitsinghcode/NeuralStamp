import torch
import torch.nn as nn
from .cnn_model import CNNWatermark
from .vit_model import ViTWatermark

class WatermarkingModel(nn.Module):
    def __init__(self):
        super(WatermarkingModel, self).__init__()
        self.cnn = CNNWatermark()
        self.vit = ViTWatermark()

        self.fc = nn.Linear(512, 150528)  # Fully connected to match output size
        self.upsample = nn.Unflatten(1, (3, 224, 224))  # Reshape to image size

    def forward(self, x):
        cnn_features = self.cnn.encoder(x)  # Get CNN features
        vit_features = self.vit(x)  # Get ViT features

        combined = torch.cat((cnn_features.flatten(1), vit_features), dim=1)  # Merge both feature sets

        output = self.fc(combined)  # [batch_size, 150528]
        output = self.upsample(output)  # [batch_size, 3, 224, 224]

        return output

if __name__ == "__main__":
    model = WatermarkingModel()
    print(model)
