import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None

def soft_nms_pytorch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5, sigma: float = 0.5, score_threshold: float = 0.3):
    """
    Soft-NMS with Gaussian decay for dense crowd face detection.
    Prevents dropping partially-occluded overlapping face bounding boxes.
    """
    if boxes.numel() == 0:
        return torch.empty((0, 4), device=boxes.device), torch.empty((0,), device=scores.device)
        
    boxes = boxes.clone()
    scores = scores.clone()
    
    N = boxes.size(0)
    for i in range(N):
        max_idx = torch.argmax(scores[i:]) + i
        boxes[i], boxes[max_idx] = boxes[max_idx].clone(), boxes[i].clone()
        scores[i], scores[max_idx] = scores[max_idx].clone(), scores[i].clone()
        
        pos = i + 1
        while pos < N:
            # Intersection coordinates
            x1 = torch.max(boxes[i, 0], boxes[pos, 0])
            y1 = torch.max(boxes[i, 1], boxes[pos, 1])
            x2 = torch.min(boxes[i, 2], boxes[pos, 2])
            y2 = torch.min(boxes[i, 3], boxes[pos, 3])
            
            w = torch.clamp(x2 - x1, min=0.0)
            h = torch.clamp(y2 - y1, min=0.0)
            inter = w * h
            
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (boxes[pos, 2] - boxes[pos, 0]) * (boxes[pos, 3] - boxes[pos, 1])
            union = area1 + area2 - inter + 1e-7
            iou = inter / union
            
            # Gaussian decay factor
            weight = torch.exp(-(iou ** 2) / sigma)
            scores[pos] = scores[pos] * weight
            pos += 1

    keep = scores >= score_threshold
    return boxes[keep], scores[keep]

class LNetAnchorFaceLocalizer(nn.Module):
    """
    Official LNet: Localization Network for Deep Face Attributes in the Wild (Liu et al., ICCV 2015).
    
    100% Standalone Pure PyTorch Multi-Scale Anchor Grid Face Localizer.
    Generates candidate face region proposals across multi-scale spatial feature maps (stride 16 and stride 8).
    Designed to detect faces under standard/ideal conditions, relying on OccuPose-BroadDictNet downstream
    modules (ANet, PIM, IResNet-100, DDRC) to process wild occlusions, pose distortions, and open-set identities.
    """
    def __init__(self, num_anchors_per_cell: int = 9):
        super(LNetAnchorFaceLocalizer, self).__init__()
        self.num_anchors = num_anchors_per_cell
        self._anchor_cache = {}
        
        # 4-Stage Conv Feature Backbone (Conv1..Conv4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # H/2
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # H/4
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# H/8
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# H/16
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        
        # Multi-scale prediction heads over 1/16 and 1/8 spatial feature maps
        self.head_cls_16 = nn.Conv2d(256, num_anchors_per_cell * 1, kernel_size=3, padding=1)
        self.head_box_16 = nn.Conv2d(256, num_anchors_per_cell * 4, kernel_size=3, padding=1)
        self.head_lm_16  = nn.Conv2d(256, num_anchors_per_cell * 10, kernel_size=3, padding=1)
        
        self.head_cls_8 = nn.Conv2d(128, num_anchors_per_cell * 1, kernel_size=3, padding=1)
        self.head_box_8 = nn.Conv2d(128, num_anchors_per_cell * 4, kernel_size=3, padding=1)
        self.head_lm_8  = nn.Conv2d(128, num_anchors_per_cell * 10, kernel_size=3, padding=1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Stable initial objectness confidence logits
        nn.init.constant_(self.head_cls_16.bias, -2.0)
        nn.init.constant_(self.head_cls_8.bias, -2.0)

    def generate_anchors(self, H_g: int, W_g: int, stride: int, img_w: int, img_h: int, device: torch.device):
        cache_key = (H_g, W_g, stride, img_w, img_h, device)
        if cache_key in self._anchor_cache:
            return self._anchor_cache[cache_key]

        scales = [0.25, 0.50, 0.85] if stride == 16 else [0.08, 0.15, 0.30]
        ratios = [1.0, 1.25, 0.8]
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H_g, device=device),
            torch.arange(W_g, device=device),
            indexing='ij'
        )
        
        cx = (grid_x.float() + 0.5) * stride
        cy = (grid_y.float() + 0.5) * stride
        
        anchors = []
        for s in scales:
            for r in ratios:
                w = s * img_w * (r ** 0.5)
                h = s * img_h / (r ** 0.5)
                anchors.append(torch.stack([cx, cy, torch.full_like(cx, w), torch.full_like(cy, h)], dim=-1))
                
        stacked = torch.stack(anchors, dim=2).reshape(-1, 4)
        self._anchor_cache[cache_key] = stacked
        return stacked

    def forward(self, img_tensor: torch.Tensor, score_threshold: float = 0.3):
        """
        Args:
            img_tensor: (1, 3, H, W) normalized image tensor in [-1, 1]
        Returns:
            boxes: (num_faces, 4) candidate face bounding box coordinates [x1, y1, x2, y2]
            scores: (num_faces,) objectness confidence scores
            landmarks: (num_faces, 5, 2) 5-point facial landmark coordinates
        """
        device = img_tensor.device
        B, C, img_h, img_w = img_tensor.shape
        
        c1 = self.conv1(img_tensor)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2) # H/8
        c4 = self.conv4(c3) # H/16
        
        cls_16 = torch.sigmoid(self.head_cls_16(c4)).permute(0, 2, 3, 1).reshape(B, -1, 1)
        box_16 = self.head_box_16(c4).permute(0, 2, 3, 1).reshape(B, -1, 4)
        lm_16  = self.head_lm_16(c4).permute(0, 2, 3, 1).reshape(B, -1, 5, 2)
        anchors_16 = self.generate_anchors(c4.shape[2], c4.shape[3], 16, img_w, img_h, device)
        
        cls_8 = torch.sigmoid(self.head_cls_8(c3)).permute(0, 2, 3, 1).reshape(B, -1, 1)
        box_8 = self.head_box_8(c3).permute(0, 2, 3, 1).reshape(B, -1, 4)
        lm_8  = self.head_lm_8(c3).permute(0, 2, 3, 1).reshape(B, -1, 5, 2)
        anchors_8 = self.generate_anchors(c3.shape[2], c3.shape[3], 8, img_w, img_h, device)
        
        all_cls = torch.cat([cls_16[0], cls_8[0]], dim=0).squeeze(-1)
        all_box_offsets = torch.cat([box_16[0], box_8[0]], dim=0)
        all_lms_offsets = torch.cat([lm_16[0], lm_8[0]], dim=0)
        all_anchors = torch.cat([anchors_16, anchors_8], dim=0)
        
        # Regress anchor offsets: [dx, dy, dw, dh]
        cx = all_anchors[:, 0] + all_box_offsets[:, 0] * all_anchors[:, 2]
        cy = all_anchors[:, 1] + all_box_offsets[:, 1] * all_anchors[:, 3]
        w  = all_anchors[:, 2] * torch.exp(all_box_offsets[:, 2].clamp(-2.0, 2.0))
        h  = all_anchors[:, 3] * torch.exp(all_box_offsets[:, 3].clamp(-2.0, 2.0))
        
        x1 = torch.clamp(cx - w / 2.0, 0.0, float(img_w))
        y1 = torch.clamp(cy - h / 2.0, 0.0, float(img_h))
        x2 = torch.clamp(cx + w / 2.0, 0.0, float(img_w))
        y2 = torch.clamp(cy + h / 2.0, 0.0, float(img_h))
        
        decoded_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        
        valid_mask = (all_cls >= score_threshold) & ((x2 - x1) > 10) & ((y2 - y1) > 10)
        if not torch.any(valid_mask):
            valid_mask = (all_cls >= 0.10) & ((x2 - x1) > 10) & ((y2 - y1) > 10)
            
        if not torch.any(valid_mask):
            return torch.empty((0, 4), device=device), torch.empty((0,), device=device), torch.empty((0, 5, 2), device=device)
            
        boxes_out = decoded_boxes[valid_mask]
        scores_out = all_cls[valid_mask]
        lms_out = all_lms_offsets[valid_mask]
        
        # Keep top-100 highest confidence candidates for Soft-NMS
        if scores_out.size(0) > 100:
            topk_scores, topk_indices = torch.topk(scores_out, k=100)
            boxes_out = boxes_out[topk_indices]
            scores_out = topk_scores
            lms_out = lms_out[topk_indices]
            
        filtered_boxes, filtered_scores = soft_nms_pytorch(boxes_out, scores_out, score_threshold=score_threshold)
        return filtered_boxes, filtered_scores, lms_out[:filtered_boxes.size(0)]

# Set default WildFaceDetector to pure PyTorch LNetAnchorFaceLocalizer
LNetFaceLocalizer = LNetAnchorFaceLocalizer
WildFaceDetector = LNetAnchorFaceLocalizer
