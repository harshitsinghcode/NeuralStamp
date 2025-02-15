import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNWatermark(nn.Module):
    def __init__(self):
        super(CNNWatermark, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, image, watermark):
        # If watermark is not in (N, C, H, W) format, reshape it.
        if watermark.ndim == 1 or watermark.ndim == 2:
            watermark = watermark.view(watermark.shape[0], 1, 1, 1)
        elif watermark.ndim == 3:
            watermark = watermark.unsqueeze(1)
            
        # Resize watermark to (224, 224) to match the input image size
        watermark_resized = F.interpolate(watermark, size=(224, 224), mode="bilinear", align_corners=False)
        # If watermark has 1 channel, expand it to 3 channels
        if watermark_resized.shape[1] == 1:
            watermark_resized = watermark_resized.expand(-1, 3, -1, -1)
        
        watermarked = self.encoder(image) + watermark_resized
        extracted = self.decoder(watermarked)
        return watermarked, extracted

if __name__ == "__main__":
    model = CNNWatermark()
    print(model)
