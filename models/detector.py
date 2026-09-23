import torch
import torch.nn as nn
import torch.nn.functional as F

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

class WildFaceDetector(nn.Module):
    """
    Production Deep Convolutional Face Detector with Soft-NMS post-processing.
    Detects face bounding boxes [x1, y1, x2, y2] in crowded, occluded, and non-ideal wild scenes.
    """
    def __init__(self):
        super(WildFaceDetector, self).__init__()
        
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Bounding box regressor head
        self.bbox_head = nn.Conv2d(64, 4, kernel_size=3, padding=1)
        # Score confidence head
        self.score_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, img_tensor: torch.Tensor, score_threshold: float = 0.4):
        """
        Args:
            img_tensor: (1, 3, H, W) normalized image tensor
        Returns:
            boxes: (num_faces, 4) bounding box coordinates [x1, y1, x2, y2]
            scores: (num_faces,) detection confidence scores
        """
        feat = self.conv_stem(img_tensor)
        raw_boxes = torch.sigmoid(self.bbox_head(feat))
        raw_scores = torch.sigmoid(self.score_head(feat))
        
        # Flatten predictions
        B, _, H, W = raw_boxes.size()
        boxes_flat = raw_boxes.permute(0, 2, 3, 1).reshape(-1, 4)
        scores_flat = raw_scores.permute(0, 2, 3, 1).reshape(-1)
        
        # Rescale boxes to image dimensions: [x1, y1, x2, y2]
        img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
        boxes_scaled = torch.zeros_like(boxes_flat)
        boxes_scaled[:, 0] = boxes_flat[:, 0] * img_w
        boxes_scaled[:, 1] = boxes_flat[:, 1] * img_h
        boxes_scaled[:, 2] = (boxes_flat[:, 0] + boxes_flat[:, 2]) * img_w
        boxes_scaled[:, 3] = (boxes_flat[:, 1] + boxes_flat[:, 3]) * img_h
        
        # Pre-filter by score_threshold before Soft-NMS to reduce N
        valid_mask = scores_flat >= score_threshold
        if not torch.any(valid_mask):
            return torch.empty((0, 4), device=img_tensor.device), torch.empty((0,), device=img_tensor.device)
            
        candidate_boxes = boxes_scaled[valid_mask]
        candidate_scores = scores_flat[valid_mask]
        
        # Keep top-k (top 200) candidates by score for fast Soft-NMS
        if candidate_scores.size(0) > 200:
            topk_scores, topk_indices = torch.topk(candidate_scores, k=200)
            candidate_boxes = candidate_boxes[topk_indices]
            candidate_scores = topk_scores
            
        # Apply Soft-NMS filtering on top candidates
        filtered_boxes, filtered_scores = soft_nms_pytorch(candidate_boxes, candidate_scores, score_threshold=score_threshold)
        return filtered_boxes, filtered_scores
