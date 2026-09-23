import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['IResNet', 'IBasicBlock', 'iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200', 'ResNet100Backbone']

class IBasicBlock(nn.Module):
    """
    Standard InsightFace / ArcFace / CurricularFace Improved Basic Block (IBasicBlock).
    Structure: BN1 -> Conv3x3 (s=1) -> BN2 -> PReLU -> Conv3x3 (s=stride) -> BN3 + Residual
    """
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('IBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in IBasicBlock")

        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class IResNet(nn.Module):
    """
    Standard InsightFace / ArcFace / CurricularFace Improved ResNet (IResNet) Architecture.
    Matches deepinsight/insightface official reference implementation.
    """
    fc_scale = 7 * 7

    def __init__(self,
                 block,
                 layers,
                 dropout: float = 0.4,
                 num_features: int = 512,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation=None,
                 fp16: bool = False):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None or 3-tuple, got {replace_stride_with_dilation}")
            
        self.groups = groups
        self.base_width = width_per_group
        
        # Stem: Conv 3 -> 64 (3x3, stride=1, padding=1, bias=False) -> BN1 -> PReLU
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)
        
        # Four Residual Stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        # Head: BN2 -> Dropout -> Flatten -> Linear -> BN3
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout > 0 else None
        self.fc = nn.Linear(512 * self.fc_scale, num_features, bias=False)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        
        # Standard InsightFace weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1, dilate: bool = False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation)
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 112, 112) normalized input face image [-1, 1]
        Returns:
            f: (B, 512) L2-normalized feature embedding vector
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        
        # Strict L2 normalization
        return F.normalize(x, p=2, dim=1)

def iresnet18(dropout=0.4, num_features=512, embedding_dim=None, **kwargs):
    if embedding_dim is not None:
        num_features = embedding_dim
    return IResNet(IBasicBlock, [2, 2, 2, 2], dropout=dropout, num_features=num_features, **kwargs)

def iresnet34(dropout=0.4, num_features=512, embedding_dim=None, **kwargs):
    if embedding_dim is not None:
        num_features = embedding_dim
    return IResNet(IBasicBlock, [3, 4, 6, 3], dropout=dropout, num_features=num_features, **kwargs)

def iresnet50(dropout=0.4, num_features=512, embedding_dim=None, **kwargs):
    if embedding_dim is not None:
        num_features = embedding_dim
    return IResNet(IBasicBlock, [3, 4, 14, 3], dropout=dropout, num_features=num_features, **kwargs)

def iresnet100(dropout=0.4, num_features=512, embedding_dim=None, **kwargs):
    if embedding_dim is not None:
        num_features = embedding_dim
    return IResNet(IBasicBlock, [3, 13, 30, 3], dropout=dropout, num_features=num_features, **kwargs)

def iresnet200(dropout=0.4, num_features=512, embedding_dim=None, **kwargs):
    if embedding_dim is not None:
        num_features = embedding_dim
    return IResNet(IBasicBlock, [6, 26, 60, 6], dropout=dropout, num_features=num_features, **kwargs)

# Alias for backwards compatibility across pipeline
ResNet100Backbone = iresnet100
