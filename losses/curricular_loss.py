import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CurricularFaceLoss(nn.Module):
    """
    CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition (Huang et al., CVPR 2020)
    
    Dynamically adjusts the relative importance of easy and hard samples during training:
    - Early training: suppresses hard negative samples to avoid divergence from noisy/corrupted faces.
    - Late training: amplifies hard misclassified samples to enforce sharp inter-class margins.
    """
    def __init__(self, in_features: int, num_classes: int, s: float = 64.0, m: float = 0.50, alpha: float = 0.99):
        super(CurricularFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.alpha = alpha
        
        # Linear classifier weights (C, d)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Exponential Moving Average parameter t
        self.register_buffer('t', torch.zeros(1))
        
        # Margin constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, in_features) L2-normalized feature embeddings
            labels: (batch_size,) ground truth class indices
        Returns:
            loss: scalar cross-entropy loss with adaptive curricular margin modulation
        """
        # Ensure L2 normalization of weights and embeddings
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        norm_weight = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity matrix: (batch_size, num_classes)
        cos_theta = F.linear(norm_embeddings, norm_weight)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        
        # Ground truth positive cosine similarities
        batch_size = embeddings.size(0)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        
        cos_yi = cos_theta[one_hot.bool()]
        
        # Update EMA curriculum parameter t during training
        if self.training:
            with torch.no_grad():
                mean_cos = cos_yi.mean()
                self.t = self.alpha * self.t + (1.0 - self.alpha) * mean_cos
        
        # Target angle margin: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        sin_yi = torch.sqrt((1.0 - cos_yi ** 2).clamp(min=1e-7))
        cos_yi_m = cos_yi * self.cos_m - sin_yi * self.sin_m
        
        # Fallback for angle overflow > pi
        cos_yi_m = torch.where(cos_yi > self.threshold, cos_yi_m, cos_yi - self.mm)
        
        # Build modulated negative similarity matrix
        # For target class (positive), use cos(theta + m)
        # For non-target classes (negatives), apply adaptive curriculum modulation N(t, cos_theta_j)
        target_margin = cos_yi_m.view(-1, 1)
        
        # Mask for non-ground truth classes
        mask = 1.0 - one_hot
        
        # Hard sample criterion: cos(theta_yi + m) < cos(theta_j)
        # i.e., negative similarity is higher than margin-adjusted ground truth similarity
        is_hard = (target_margin < cos_theta) & mask.bool()
        
        # Modulation function N(t, cos_theta_j) = cos_theta_j * (t + cos_theta_j) for hard negatives
        modulated_negatives = torch.where(
            is_hard,
            cos_theta * (self.t + cos_theta),
            cos_theta
        )
        
        # Combine target positive logits and modulated negative logits
        output_logits = torch.where(one_hot.bool(), target_margin, modulated_negatives)
        output_logits = output_logits * self.s
        
        loss = F.cross_entropy(output_logits, labels)
        return loss
