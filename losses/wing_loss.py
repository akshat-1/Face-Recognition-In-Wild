import math
import torch
import torch.nn as nn

class WingLoss(nn.Module):
    """
    Official Wing Loss for Facial Landmark Localization (Feng et al., CVPR 2018 / arXiv:1711.06753).
    
    Amplifies gradients for small landmark localization errors:
    Wing(x) = w * ln(1 + |x| / epsilon)  if |x| < w else |x| - C
    where C = w - w * ln(1 + w / epsilon).
    """
    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.c = w - w * math.log(1.0 + w / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, N, 2) predicted landmark coordinates
            target: (B, N, 2) ground-truth landmark coordinates
        Returns:
            loss: scalar landmark regression loss
        """
        diff = pred - target
        abs_diff = torch.abs(diff)
        
        flag = (abs_diff < self.w).float()
        loss = flag * (self.w * torch.log(1.0 + abs_diff / self.epsilon)) + (1.0 - flag) * (abs_diff - self.c)
        return torch.mean(loss)
