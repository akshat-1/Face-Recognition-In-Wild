import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class BroadFaceCurricularLoss(nn.Module):
    """
    Official Kakao Enterprise BroadFace + CurricularFace PyTorch Implementation.
    Ref: https://github.com/kakaoenterprise/BroadFace
    "BroadFace: Looking at Tens of Thousands of People at Once for Face Recognition", ECCV 2020.
    
    Combines BroadFace memory queue proxy compensation with CurricularFace adaptive margin learning.
    """
    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 100,
        scale_factor: float = 64.0,
        margin: float = 0.50,
        alpha: float = 0.99,
        queue_size: int = 10000,
        compensate: bool = True,
        feature_dim: int = None,
    ):
        super(BroadFaceCurricularLoss, self).__init__()
        if feature_dim is not None:
            in_features = feature_dim
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale_factor = scale_factor
        self.margin = margin
        self.alpha = alpha
        self.queue_size = queue_size
        self.compensate = compensate

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.register_buffer('t', torch.zeros(1))
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        feature_mb = torch.zeros(0, in_features)
        label_mb = torch.zeros(0, dtype=torch.int64)
        proxy_mb = torch.zeros(0, in_features)
        self.register_buffer("feature_mb", feature_mb)
        self.register_buffer("label_mb", label_mb)
        self.register_buffer("proxy_mb", proxy_mb)

    @property
    def queue_ptr(self) -> torch.Tensor:
        return torch.tensor(self.feature_mb.shape[0], device=self.weight.device)

    @property
    def is_full(self) -> torch.Tensor:
        return torch.tensor(self.feature_mb.shape[0] >= self.queue_size, device=self.weight.device)

    def get_queue_samples(self):
        return self.feature_mb, self.label_mb

    @torch.no_grad()
    def update(self, input_tensor: torch.Tensor, label: torch.Tensor):
        label = torch.clamp(label.long(), 0, self.num_classes - 1)
        self.feature_mb = torch.cat([self.feature_mb, input_tensor.detach()], dim=0)
        self.label_mb = torch.cat([self.label_mb, label.detach()], dim=0)
        self.proxy_mb = torch.cat(
            [self.proxy_mb, self.weight.data[label].clone()], dim=0
        )

        over_size = self.feature_mb.shape[0] - self.queue_size
        if over_size > 0:
            self.feature_mb = self.feature_mb[over_size:]
            self.label_mb = self.label_mb[over_size:]
            self.proxy_mb = self.proxy_mb[over_size:]

    def compute_curricular(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        y = torch.clamp(y.long(), 0, self.num_classes - 1)
        norm_embeddings = F.normalize(x.float(), p=2, dim=1)
        norm_weight = F.normalize(w.float(), p=2, dim=1)
        
        cos_theta = F.linear(norm_embeddings, norm_weight)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1.0)
        
        cos_yi = cos_theta[one_hot.bool()]
        
        if self.training:
            with torch.no_grad():
                mean_cos = cos_yi.mean()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(mean_cos, op=dist.ReduceOp.SUM)
                    mean_cos /= dist.get_world_size()
                self.t = self.alpha * self.t + (1.0 - self.alpha) * mean_cos
        
        sin_yi = torch.sqrt((1.0 - cos_yi ** 2).clamp(min=1e-7))
        cos_yi_m = cos_yi * self.cos_m - sin_yi * self.sin_m
        cos_yi_m = torch.where(cos_yi > self.th, cos_yi_m, cos_yi - self.mm)
        
        target_margin = cos_yi_m.view(-1, 1)
        mask = 1.0 - one_hot
        
        is_hard = (target_margin < cos_theta) & mask.bool()
        modulated_negatives = torch.where(
            is_hard,
            cos_theta * (self.t + cos_theta),
            cos_theta
        )
        
        output_logits = torch.where(one_hot.bool(), target_margin, modulated_negatives)
        output_logits = output_logits * self.scale_factor
        
        return F.cross_entropy(output_logits, y)

    def forward(self, input_tensor: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if self.label_mb.shape[0] == 0:
            batch_loss = self.compute_curricular(input_tensor, label, self.weight)
            with torch.no_grad():
                self.update(input_tensor, label)
            return batch_loss

        weight_now = self.weight.data[self.label_mb]
        delta_weight = weight_now - self.proxy_mb

        if self.compensate:
            update_feature_mb = (
                self.feature_mb
                + (
                    self.feature_mb.norm(p=2, dim=1, keepdim=True)
                    / (self.proxy_mb.norm(p=2, dim=1, keepdim=True) + 1e-7)
                )
                * delta_weight
            )
        else:
            update_feature_mb = self.feature_mb

        large_input = torch.cat([update_feature_mb, input_tensor], dim=0)
        large_label = torch.cat([self.label_mb, label], dim=0)

        batch_loss = self.compute_curricular(input_tensor, label, self.weight)
        broad_loss = self.compute_curricular(large_input, large_label, self.weight)
        
        with torch.no_grad():
            self.update(input_tensor, label)

        return batch_loss + broad_loss

# Backward compatibility alias
BroadFaceMemoryQueue = BroadFaceCurricularLoss
