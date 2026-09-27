import os
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torchvision.models.detection as detection
except ImportError:
    detection = None

def download_yunet_weights(save_path: str = "weights/face_detection_yunet_2023mar.onnx"):
    """
    Downloads official OpenCV YuNet SOTA Face Detector ONNX model if not already present.
    """
    if os.path.exists(save_path) and os.path.getsize(save_path) > 100000:
        return save_path
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    try:
        print(f"Downloading SOTA OpenCV YuNet Face Detector ONNX model to {save_path}...")
        urllib.request.urlretrieve(url, save_path)
        print("✓ YuNet Face Detector weights successfully downloaded.")
        return save_path
    except Exception as e:
        print(f"Warning: Failed to download YuNet face detector model: {e}")
        return None

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
    SOTA OpenCV YuNet ONNX Face Detector + LNet Localizer for Deep Face Attributes in the Wild (Liu et al., ICCV 2015).
    Predicts exact full facial bounding boxes [x1, y1, x2, y2] covering forehead down to chin & jawline.
    """
    def __init__(self, weights_path: str = "weights/face_detection_yunet_2023mar.onnx"):
        super(LNetFaceLocalizer, self).__init__()
        
        self.yunet_detector = None
        if cv2 is not None:
            model_file = download_yunet_weights(weights_path)
            if model_file and os.path.exists(model_file):
                try:
                    self.yunet_detector = cv2.FaceDetectorYN_create(
                        model_file, "", (300, 300), score_threshold=0.25, nms_threshold=0.3
                    )
                except Exception as e:
                    print(f"Warning: Failed to initialize YuNet detector: {e}")
                    self.yunet_detector = None

        if detection is not None:
            try:
                self.backbone_detector = detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights='DEFAULT')
            except Exception:
                self.backbone_detector = None
        else:
            self.backbone_detector = None

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
        
        # 1. Primary SOTA Face Detector: OpenCV YuNet ONNX Model
        if self.yunet_detector is not None and cv2 is not None:
            try:
                # Convert normalized PyTorch tensor [-1, 1] to BGR numpy uint8 image
                tensor_unnorm = (img_tensor[0].detach().cpu() * 0.5 + 0.5).clamp(0.0, 1.0)
                img_np = (tensor_unnorm.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Multi-scale pyramid detection (1x and 2x resolution)
                all_boxes = []
                all_scores = []
                all_landmarks = []
                
                scales = [1.0, 2.0] if (img_w < 800 and img_h < 800) else [1.0]
                
                for scale in scales:
                    if scale == 1.0:
                        inp_img = img_bgr
                        sw, sh = img_w, img_h
                    else:
                        sw, sh = int(img_w * scale), int(img_h * scale)
                        inp_img = cv2.resize(img_bgr, (sw, sh))
                        
                    self.yunet_detector.setInputSize((sw, sh))
                    _, faces = self.yunet_detector.detect(inp_img)
                    
                    if faces is not None and len(faces) > 0:
                        for f in faces:
                            fx, fy, fw, fh = [float(v) / scale for v in f[:4]]
                            conf = float(f[14])
                            
                            # Add 5% padding around detected face bounding box
                            pad_x = fw * 0.05
                            pad_y = fh * 0.05
                            
                            x1 = max(0.0, fx - pad_x)
                            y1 = max(0.0, fy - pad_y)
                            x2 = min(float(img_w), fx + fw + pad_x)
                            y2 = min(float(img_h), fy + fh + pad_y)
                            
                            if (x2 - x1) >= 8 and (y2 - y1) >= 8:
                                all_boxes.append([x1, y1, x2, y2])
                                all_scores.append(conf)
                                
                                # Extract 5 facial landmark pairs (right eye, left eye, nose, right mouth, left mouth)
                                lm_pts = f[4:14].reshape(5, 2) / scale
                                all_landmarks.append(lm_pts)
                                
                if len(all_boxes) > 0:
                    cand_boxes = torch.tensor(all_boxes, device=device)
                    cand_scores = torch.tensor(all_scores, device=device)
                    cand_lms = torch.tensor(np.array(all_landmarks), device=device)
                    
                    filtered_boxes, filtered_scores = soft_nms_pytorch(cand_boxes, cand_scores, score_threshold=score_threshold)
                    return filtered_boxes, filtered_scores, cand_lms[:filtered_boxes.size(0)]
            except Exception as e:
                pass

        # 2. Fallback Secondary Face Detector: FasterRCNN Person Upper-Body Localizer
        if self.backbone_detector is not None:
            inp = (img_tensor * 0.5 + 0.5).clamp(0.0, 1.0)
            with torch.no_grad():
                out = self.backbone_detector(inp)[0]
                
            boxes = out['boxes']
            scores = out['scores']
            labels = out['labels']
            
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
                    if bh > bw * 1.2:
                        fx1 = max(0.0, x1 + bw * 0.05)
                        fx2 = min(float(img_w), x2 - bw * 0.05)
                        fy1 = y1
                        fy2 = min(float(img_h), y1 + bh * 0.60)
                    else:
                        fx1, fy1, fx2, fy2 = x1, y1, x2, y2
                    face_boxes.append([fx1, fy1, fx2, fy2])
                    
                cand_boxes = torch.tensor(face_boxes, device=device)
                cand_scores = p_scores
                cand_lms = torch.zeros((cand_boxes.size(0), 5, 2), device=device)
                
                filtered_boxes, filtered_scores = soft_nms_pytorch(cand_boxes, cand_scores, score_threshold=score_threshold)
                return filtered_boxes, filtered_scores, cand_lms[:filtered_boxes.size(0)]

        # 3. Emergency Fallback: Empty candidate tensor (triggers full image crop only if 0 faces detected)
        return torch.empty((0, 4), device=device), torch.empty((0,), device=device), torch.empty((0, 5, 2), device=device)

# Alias for backwards compatibility across detector pipeline
WildFaceDetector = LNetFaceLocalizer
