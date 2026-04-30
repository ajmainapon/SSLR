import torch.nn as nn
import torch.nn.functional as F


class LinearSegHead(nn.Module):
    """Linear probe: 1x1 conv on patch tokens + bilinear upsample."""
    def __init__(self, dim=768, num_classes=11, patch_size=16, img_size=224):
        super().__init__()
        self.side = img_size // patch_size
        self.img_size = img_size
        self.classifier = nn.Conv2d(dim, num_classes, 1)

    def forward(self, tokens):
        B, N, D = tokens.shape
        z = tokens.transpose(1, 2).reshape(B, D, self.side, self.side)
        z = self.classifier(z)
        return F.interpolate(z, size=(self.img_size, self.img_size),
                             mode="bilinear", align_corners=False)


class ConvSegHead(nn.Module):
    """Two 3x3 conv blocks + bilinear upsample. Use if linear underperforms."""
    def __init__(self, dim=768, num_classes=11, patch_size=16, img_size=224, hidden=256):
        super().__init__()
        self.side = img_size // patch_size
        self.img_size = img_size
        self.head = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, num_classes, 1),
        )

    def forward(self, tokens):
        B, N, D = tokens.shape
        z = tokens.transpose(1, 2).reshape(B, D, self.side, self.side)
        z = self.head(z)
        return F.interpolate(z, size=(self.img_size, self.img_size),
                             mode="bilinear", align_corners=False)
