import torch
import torch.nn as nn
import torch.nn.functional as F

class IResNetBlock(nn.Module):
    """
    Improved ResNet (IResNet) Residual Block for Deep Face Recognition (ArcFace / CurricularFace).
    Structure: BN1 -> Conv3x3 -> BN2 -> PReLU -> Conv3x3 -> BN3 + Residual
    """
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(IResNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        return out + residual

class ResNet100Backbone(nn.Module):
    """
    SOTA Improved ResNet-100 (IResNet-100) Identity Embedding Extractor for Deep Face Recognition.
    Block stages: [3, 13, 30, 3] with 512-d L2-normalized feature output.
    """
    def __init__(self, embedding_dim: int = 512, layers: tuple = (3, 13, 30, 3), fp16: bool = False):
        super(ResNet100Backbone, self).__init__()
        self.in_planes = 64
        self.embedding_dim = embedding_dim
        self.fp16 = fp16

        # Stem: Conv3x3 64 -> BN -> PReLU
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        # Stage 1: 64 planes, stride 2 downsampling on 1st block
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        # Stage 2: 128 planes, stride 2 downsampling on 1st block
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        # Stage 3: 256 planes, stride 2 downsampling on 1st block
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        # Stage 4: 512 planes, stride 2 downsampling on 1st block
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        # Feature Output FC Head
        self.fc_head = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(IResNetBlock(self.in_planes, planes, stride))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(IResNetBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 112, 112) normalized input face image [-1, 1]
        Returns:
            f: (B, 512) L2-normalized identity feature vector
        """
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        embeddings = self.fc_head(out)
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        return norm_embeddings
