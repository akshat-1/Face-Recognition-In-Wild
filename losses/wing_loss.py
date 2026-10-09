import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WingLoss(nn.Module):
    """
    Official Wing Loss for Deep Face Alignment (Feng et al., CVPR 2018 / arXiv:1711.06753).
    
    Mathematical Formulation (Equation 4):
    Wing(x) = w * ln(1 + |x| / epsilon)            if |x| < w
              |x| - C                              if |x| >= w
    where C = w - w * ln(1 + w / epsilon).
    
    Official paper parameters: w = 10.0, epsilon = 2.0.
    """
    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        super(WingLoss, self).__init__()
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.c = self.omega - self.omega * math.log(1.0 + self.omega / self.epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, N, 2) or arbitrary shape predicted landmark coordinates
            target: (B, N, 2) or matching shape ground-truth landmark coordinates
        Returns:
            loss: scalar landmark regression loss
        """
        delta_x = pred - target
        abs_delta_x = torch.abs(delta_x)
        
        # Smooth step boolean mask for non-linear vs linear region (|x| < w)
        smooth_mask = (abs_delta_x < self.omega).to(dtype=pred.dtype)
        
        # Non-linear logarithmic region: w * ln(1 + |x| / epsilon)
        nonlinear_loss = self.omega * torch.log1p(abs_delta_x / self.epsilon)
        
        # Linear region: |x| - C
        linear_loss = abs_delta_x - self.c
        
        # Element-wise loss selection
        loss = smooth_mask * nonlinear_loss + (1.0 - smooth_mask) * linear_loss
        return torch.mean(loss)

class AdaptiveWingLoss(nn.Module):
    """
    Official Adaptive Wing Loss for Robust Facial Landmark Localization (Wang et al., ICCV 2019 / arXiv:1904.07399).
    
    Mathematical Formulation (Equation 7):
    AWing(y, y_hat) = omega * ln(1 + (|y - y_hat| / epsilon)^(alpha - y))  if |y - y_hat| < theta
                      A * |y - y_hat| - C                                  otherwise
                      
    where:
    A = omega * (1 / (1 + (theta / epsilon)^(alpha - y))) * (alpha - y) * (theta / epsilon)^(alpha - y - 1) * (1 / epsilon)
    C = (theta * A) - omega * ln(1 + (theta / epsilon)^(alpha - y))
    
    Official paper parameters: omega = 14.0, alpha = 2.1, float_epsilon = 1.0, theta = 0.5.
    """
    def __init__(self, omega: float = 14.0, alpha: float = 2.1, epsilon: float = 1.0, theta: float = 0.5):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.theta = float(theta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) or (B, N, 2) predicted landmark heatmaps/coordinates
            target: (B, C, H, W) or (B, N, 2) ground-truth landmark heatmaps/coordinates
        Returns:
            loss: scalar adaptive wing loss
        """
        delta_y = target - pred
        abs_delta_y = torch.abs(delta_y)
        
        # Smooth mask for non-linear logarithmic vs linear boundary (|y - y_hat| < theta)
        mask = (abs_delta_y < self.theta).to(dtype=pred.dtype)
        
        # Adaptive exponent: alpha - y
        exponent = self.alpha - target
        
        # Non-linear logarithmic term: omega * ln(1 + (|y - y_hat| / epsilon)^(alpha - y))
        ratio = abs_delta_y / self.epsilon
        nonlinear_term = self.omega * torch.log1p(torch.pow(ratio.clamp(min=1e-8), exponent))
        
        # Derivatives for linear continuous matching
        term1 = 1.0 / (1.0 + torch.pow(self.theta / self.epsilon, exponent))
        term2 = exponent * torch.pow(self.theta / self.epsilon, exponent - 1.0) * (1.0 / self.epsilon)
        A = self.omega * term1 * term2
        C = (self.theta * A) - self.omega * torch.log1p(torch.pow(self.theta / self.epsilon, exponent))
        
        linear_term = A * abs_delta_y - C
        
        loss = mask * nonlinear_term + (1.0 - mask) * linear_term
        return torch.mean(loss)
