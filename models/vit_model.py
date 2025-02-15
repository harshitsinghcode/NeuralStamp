from transformers import ViTModel
import torch.nn as nn
import torch

class ViTModelWrapper(nn.Module):
    def __init__(self):
        super(ViTModelWrapper, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.fc = nn.Linear(self.vit.config.hidden_size, 128)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        x = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x
