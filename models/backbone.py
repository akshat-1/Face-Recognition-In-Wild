import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu1 = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.prelu2 = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.prelu2(out + residual)
        return out

class ResNet100Backbone(nn.Module):
    """
    Deep ResNet Identity Embedding Extractor for Face Recognition in the Wild.
    Maps 112x112 face crop images to a 512-dimensional L2-normalized feature embedding vector f.
    Supports lightweight ResNet (3, 4, 6, 3) for fast CPU inference and full ResNet-100 (3, 13, 30, 3) for high-capacity training.
    """
    def __init__(self, embedding_dim: int = 512, layers: tuple = (3, 4, 6, 3)):
        super(ResNet100Backbone, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Stem
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        
        # Stage 1: 64 channels
        self.layer1 = self._make_layer(64, num_blocks=layers[0])
        # Stage 2: 128 channels
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2 = self._make_layer(128, num_blocks=layers[1])
        # Stage 3: 256 channels
        self.down2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3 = self._make_layer(256, num_blocks=layers[2])
        # Stage 4: 512 channels
        self.down3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer4 = self._make_layer(512, num_blocks=layers[3])
        
        # Output embedding head
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )

    def _make_layer(self, channels: int, num_blocks: int) -> nn.Sequential:
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResNetBlock(channels))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 3, 112, 112) normalized input face image [-1, 1]
        Returns:
            f: (batch_size, 512) L2-normalized identity embedding vector
        """
        out = self.input_layer(x)
        out = self.layer1(out)
        out = self.down1(out)
        out = self.layer2(out)
        out = self.down2(out)
        out = self.layer3(out)
        out = self.down3(out)
        out = self.layer4(out)
        
        embeddings = self.output_layer(out)
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        return norm_embeddings
