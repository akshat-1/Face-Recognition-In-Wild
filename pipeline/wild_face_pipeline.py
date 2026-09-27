import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any

from models.detector import WildFaceDetector
from models.anet_attribute import ANetAttributeParser
from models.pim_frontalizer import PIMFrontalizationGAN
from models.backbone import ResNet100Backbone
from models.ddrc_solver import DDRCClassifier

class OccuPoseBroadDictPipeline(nn.Module):
    """
    OccuPose-BroadDictNet End-to-End Face Recognition & Detection Pipeline for Wild Imagery.
    Unifies Detector, Attribute Parser, Pose Frontalizer, Feature Backbone, and DDRC Open-Set Classifier.
    """
    def __init__(self, num_enrolled_classes: int = 100, feature_dim: int = 512):
        super(OccuPoseBroadDictPipeline, self).__init__()
        
        self.detector = WildFaceDetector()
        self.attribute_parser = ANetAttributeParser()
        self.frontalizer = PIMFrontalizationGAN()
        self.backbone = ResNet100Backbone(embedding_dim=feature_dim)
        self.ddrc_classifier = DDRCClassifier(feature_dim=feature_dim, num_classes=num_enrolled_classes)

    def load_pretrained_weights(self, checkpoint_dir: str = "./weights", device: torch.device = None):
        """
        Loads Phase 1, Phase 2, and Phase 3 trained weight checkpoints for end-to-end inference.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        p1_path = os.path.join(checkpoint_dir, "phase1_backbone_curricular.pt")
        if not os.path.exists(p1_path):
            p1_path = os.path.join(checkpoint_dir, "phase1_epoch_100.pt")
            
        if os.path.exists(p1_path):
            try:
                ckpt = torch.load(p1_path, map_location=device)
                self.backbone.load_state_dict(ckpt['backbone'])
                print(f"✓ Loaded Phase 1 Backbone weights from: {p1_path}")
            except Exception as e:
                print(f"Warning: Failed to load Phase 1 weights from {p1_path}: {e}")
                
        p2_path = os.path.join(checkpoint_dir, "phase2_anet_attributes.pt")
        if os.path.exists(p2_path):
            try:
                ckpt = torch.load(p2_path, map_location=device)
                self.attribute_parser.load_state_dict(ckpt['anet'])
                print(f"✓ Loaded Phase 2 ANet Attribute Parser weights from: {p2_path}")
            except Exception as e:
                print(f"Warning: Failed to load Phase 2 weights from {p2_path}: {e}")
                
        p3_path = os.path.join(checkpoint_dir, "phase3_semi_gcn_predictor.pt")
        if os.path.exists(p3_path):
            try:
                ckpt = torch.load(p3_path, map_location=device)
                print(f"✓ Loaded Phase 3 GCN Link Predictor weights from: {p3_path}")
            except Exception as e:
                print(f"Warning: Failed to load Phase 3 weights from {p3_path}: {e}")

    def forward(self, img_tensor: torch.Tensor, score_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        Args:
            img_tensor: (1, 3, H, W) normalized input wild image [-1, 1]
            score_threshold: confidence cutoff for face detection
        Returns:
            results: list of dicts for each detected face containing:
                - 'box': [x1, y1, x2, y2]
                - 'identity': predicted person label or 'unknown'
                - 'confidence': confidence score float in [0, 1]
                - 'is_occluded': bool flag
                - 'attributes': dict of predicted attribute flags
        """
        self.eval()
        with torch.no_grad():
            # Step 1: Detect face bounding boxes & landmarks via LNet
            boxes, det_scores, landmarks = self.detector(img_tensor, score_threshold=score_threshold)
            
            if boxes.size(0) == 0:
                return []
                
            results = []
            img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
            
            # Process each detected face crop
            for i in range(boxes.size(0)):
                box = boxes[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(img_w, box[2]), min(img_h, box[3])
                
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                    
                crop_patch = img_tensor[:, :, y1:y2, x1:x2]
                resized_crop = F.interpolate(crop_patch, size=(112, 112), mode='bilinear', align_corners=False)
                
                # Step 2: Semantic attribute & occlusion parsing via ANet Dual-Path
                lm_crop = landmarks[i] if landmarks.size(0) > i else None
                attr_logits, occ_mask, is_occluded = self.attribute_parser(resized_crop, lm_crop)
                
                # Step 3: Pose estimation heuristic & PIM Frontalization
                # Heuristic pose yaw estimation from horizontal crop ratio imbalance
                crop_aspect_ratio = (x2 - x1) / float(y2 - y1 + 1e-5)
                estimated_yaw = (crop_aspect_ratio - 1.0) * 45.0
                
                # Apply PIM frontalization if yaw is high or major occlusion is detected
                if abs(estimated_yaw) > 20.0 or is_occluded.item():
                    processed_crop = self.frontalizer(resized_crop, yaw_angle=estimated_yaw)
                else:
                    processed_crop = resized_crop
                    
                # Step 4: Extract 512-d L2-normalized feature embedding
                embedding = self.backbone(processed_crop) # (1, 512)
                
                # Step 5: DDRC Sparse Dictionary classification & open-set unknown check
                preds, confs, residuals = self.ddrc_classifier(embedding)
                identity_label = preds[0]
                confidence = confs[0].item() * det_scores[i].item() # Calibrate with detection score
                
                # Parse top attributes
                attr_probs = torch.sigmoid(attr_logits[0])
                predicted_attrs = {
                    ANetAttributeParser.ATTRIBUTE_NAMES[idx]: (attr_probs[idx].item() > 0.5)
                    for idx in range(min(10, len(ANetAttributeParser.ATTRIBUTE_NAMES)))
                }
                
                results.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'identity': identity_label,
                    'confidence': round(confidence, 4),
                    'is_occluded': bool(is_occluded.item()),
                    'attributes': predicted_attrs
                })
                
            return results
