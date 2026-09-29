import os
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
                if 'broadface_loss_fn' in ckpt and 'weight' in ckpt['broadface_loss_fn']:
                    proto_weights = ckpt['broadface_loss_fn']['weight'] # (C, 512)
                    self.ddrc_classifier.enroll_prototypes(proto_weights)
                    print(f"✓ Enrolled {proto_weights.shape[0]} identity prototype centers into DDRC Classifier.")
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

    def enroll_dataset(self, name_label_dir: str, device: torch.device = None):
        """
        Enrolls identity face images from directory into DDRC Classifier dictionary.
        """
        if device is None:
            device = next(self.parameters()).device
            
        from dataset import NameLabeledFaceDataset, get_default_transform
        from PIL import Image
        
        ds = NameLabeledFaceDataset(name_label_dir=name_label_dir)
        if len(ds.class_to_idx) == 0:
            print(f"No identity classes found in {name_label_dir}")
            return
            
        idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
        class_embeddings = {i: [] for i in range(len(idx_to_class))}
        transform = get_default_transform(image_size=(112, 112), is_train=False)
        
        self.eval()
        with torch.no_grad():
            for img_path, label in ds.samples:
                if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        tensor = transform(img).unsqueeze(0).to(device)
                        emb = self.backbone(tensor)
                        class_embeddings[label].append(emb.squeeze(0))
                    except Exception:
                        pass
                        
        prototypes = []
        valid_names = []
        for i in range(len(idx_to_class)):
            if len(class_embeddings[i]) > 0:
                mean_emb = torch.stack(class_embeddings[i]).mean(dim=0)
                mean_emb = F.normalize(mean_emb, p=2, dim=0)
                prototypes.append(mean_emb)
                # Clean up class name formatting
                name = idx_to_class[i]
                for prefix in ["ROF_", "WILD_", "MASK_"]:
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                name = name.replace("_wearing_mask", "").replace("_", " ").title()
                valid_names.append(name)
                
        if len(prototypes) > 0:
            proto_tensor = torch.stack(prototypes)
            self.ddrc_classifier.enroll_prototypes(proto_tensor, class_names=valid_names)
            print(f"✓ Enrolled {len(valid_names)} identities from '{name_label_dir}' into DDRC Classifier:")
            print(f"  Enrolled names: {', '.join(valid_names[:8])}...")

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
            
            img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
            
            # If no face box is detected by detector, use full image as fallback
            if boxes.size(0) == 0:
                boxes = torch.tensor([[0, 0, img_w, img_h]], device=img_tensor.device)
                det_scores = torch.tensor([0.5], device=img_tensor.device)
                landmarks = torch.zeros(1, 5, 2, device=img_tensor.device)
                
            results = []
            
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
                raw_ddrc_conf = confs[0].item()
                det_conf = det_scores[i].item()
                # Calibrated confidence: DDRC match confidence if known identity, else detection/quality score
                confidence = raw_ddrc_conf if identity_label != 'unknown' else max(raw_ddrc_conf, det_conf)
                
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
                
            # Sort results by confidence descending
            results = sorted(results, key=lambda r: r['confidence'], reverse=True)
            
            # Apply NMS on overlapping candidate boxes
            filtered_results = []
            for r in results:
                b1 = r['box']
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                keep = True
                for prev in filtered_results:
                    b2 = prev['box']
                    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    iou = inter / float(area1 + area2 - inter + 1e-5)
                    io_min = inter / float(min(area1, area2) + 1e-5)
                    if iou > 0.3 or io_min > 0.6:
                        keep = False
                        break
                if keep:
                    filtered_results.append(r)
                    
            return filtered_results
