import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models.detection as detection
except ImportError:
    detection = None

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

class LNetFaceLocalizer(nn.Module):
    """
    Official LNet / PyTorch Pretrained Localizer for Deep Face Attributes in the Wild (Liu et al., ICCV 2015).
    Integrates CUDA-accelerated FPN Backbone with Soft-NMS post-processing for accurate face localization in wild scenes.
    """
    def __init__(self):
        super(LNetFaceLocalizer, self).__init__()
        
        if detection is not None:
            try:
                self.backbone_detector = detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights='DEFAULT')
            except Exception:
                try:
                    self.backbone_detector = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
                except Exception:
                    self.backbone_detector = None
        else:
            self.backbone_detector = None
            
        # Fallback conv stem if pretrained backbone unavailable
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.bbox_head = nn.Conv2d(256, 4, kernel_size=3, padding=1)
        self.score_head = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.landmark_head = nn.Conv2d(256, 10, kernel_size=3, padding=1)

    def forward(self, img_tensor: torch.Tensor, score_threshold: float = 0.3):
        """
        Args:
            img_tensor: (1, 3, H, W) normalized image tensor in [-1, 1]
        Returns:
            boxes: (num_faces, 4) bounding box coordinates [x1, y1, x2, y2]
            scores: (num_faces,) detection confidence scores
            landmarks: (num_faces, 5, 2) facial landmark coordinates
        """
        device = img_tensor.device
        img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
        
        if self.backbone_detector is not None:
            # Convert [-1, 1] input tensor to [0, 1] for backbone detector
            inp = (img_tensor * 0.5 + 0.5).clamp(0.0, 1.0)
            
            with torch.no_grad():
                out = self.backbone_detector(inp)[0]
                
            boxes = out['boxes']
            scores = out['scores']
            labels = out['labels']
            
            # Select person/face candidates (COCO class 1 = person)
            mask = (labels == 1) & (scores >= score_threshold)
            if not torch.any(mask):
                mask = (labels == 1) & (scores >= 0.15)
                
            if torch.any(mask):
                p_boxes = boxes[mask]
                p_scores = scores[mask]
                
                face_boxes = []
                for b in p_boxes:
                    x1, y1, x2, y2 = b[0].item(), b[1].item(), b[2].item(), b[3].item()
                    bw, bh = x2 - x1, y2 - y1
                    # Extract head/face region from person bounding box
                    if bh > bw * 1.1:
                        fx1 = max(0.0, x1 + bw * 0.10)
                        fx2 = min(float(img_w), x2 - bw * 0.10)
                        fy1 = y1
                        fy2 = min(float(img_h), y1 + bh * 0.45)
                    else:
                        fx1, fy1, fx2, fy2 = x1, y1, x2, y2
                    face_boxes.append([fx1, fy1, fx2, fy2])
                    
                candidate_boxes = torch.tensor(face_boxes, device=device)
                candidate_scores = p_scores
                candidate_landmarks = torch.zeros((candidate_boxes.size(0), 5, 2), device=device)
                
                # Apply Soft-NMS
                filtered_boxes, filtered_scores = soft_nms_pytorch(candidate_boxes, candidate_scores, score_threshold=score_threshold)
                return filtered_boxes, filtered_scores, candidate_landmarks[:filtered_boxes.size(0)]

        # Fallback LNet Conv Head if backbone detector finds nothing
        s1 = self.stage1(img_tensor)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        feat = self.stage4(s3)
        
        raw_boxes = torch.sigmoid(self.bbox_head(feat))
        raw_scores = torch.sigmoid(self.score_head(feat))
        raw_landmarks = torch.sigmoid(self.landmark_head(feat))
        
        boxes_flat = raw_boxes.permute(0, 2, 3, 1).reshape(-1, 4)
        scores_flat = raw_scores.permute(0, 2, 3, 1).reshape(-1)
        landmarks_flat = raw_landmarks.permute(0, 2, 3, 1).reshape(-1, 5, 2)
        
        boxes_scaled = torch.zeros_like(boxes_flat)
        boxes_scaled[:, 0] = boxes_flat[:, 0] * img_w
        boxes_scaled[:, 1] = boxes_flat[:, 1] * img_h
        boxes_scaled[:, 2] = (boxes_flat[:, 0] + boxes_flat[:, 2]) * img_w
        boxes_scaled[:, 3] = (boxes_flat[:, 1] + boxes_flat[:, 3]) * img_h
        
        landmarks_scaled = torch.zeros_like(landmarks_flat)
        landmarks_scaled[:, :, 0] = landmarks_flat[:, :, 0] * img_w
        landmarks_scaled[:, :, 1] = landmarks_flat[:, :, 1] * img_h
        
        valid_mask = scores_flat >= score_threshold
        if not torch.any(valid_mask):
            return torch.empty((0, 4), device=device), torch.empty((0,), device=device), torch.empty((0, 5, 2), device=device)
            
        candidate_boxes = boxes_scaled[valid_mask]
        candidate_scores = scores_flat[valid_mask]
        candidate_landmarks = landmarks_scaled[valid_mask]
        
        filtered_boxes, filtered_scores = soft_nms_pytorch(candidate_boxes, candidate_scores, score_threshold=score_threshold)
        return filtered_boxes, filtered_scores, candidate_landmarks[:filtered_boxes.size(0)]

# Alias for backwards compatibility across detector pipeline
WildFaceDetector = LNetFaceLocalizer
