import torch
import torch.nn as nn
from .cnn_model import CNNModel
from .vit_model import ViTModelWrapper

class WatermarkingModel(nn.Module):
    def __init__(self):
        super(WatermarkingModel, self).__init__()
        self.cnn = CNNModel()
        self.vit = ViTModelWrapper()
        self.fc = nn.Linear(256, 1)  # Combining CNN & ViT features

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.vit(x)
        combined = torch.cat((cnn_features, vit_features), dim=1)
        output = self.fc(combined)
        return output
