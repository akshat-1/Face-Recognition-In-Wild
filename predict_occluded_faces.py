import os
import argparse
import csv
import time
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from pipeline.wild_face_pipeline import OccuPoseBroadDictPipeline
from models.anet_attribute import ANetAttributeParser
from dataset import get_default_transform

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def preprocess_image(image_path: str, image_size=(224, 224)):
    """
    Loads and preprocesses an image file into a normalized tensor [-1, 1].
    """
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    
    transform = get_default_transform(image_size=image_size, is_train=False)
    img_tensor = transform(img).unsqueeze(0) # (1, 3, H, W)
    return img, img_tensor, (orig_w, orig_h)

def analyze_occlusion_mask(occ_mask: torch.Tensor):
    """
    Analyzes the 7x7 spatial occlusion map M_spatial returned by ANet.
    Returns occlusion severity percentage and spatial zone breakdown.
    """
    mask_grid = occ_mask.squeeze().detach().cpu().numpy() # (7, 7) array in [0, 1]
    
    occlusion_severity = (1.0 - np.mean(mask_grid)) * 100.0
    
    upper_face_score = np.mean(mask_grid[:3, :]) # Top 3 rows (Eyes/Hat/Glasses)
    lower_face_score = np.mean(mask_grid[3:, :]) # Bottom 4 rows (Nose/Mouth/Mask)
    
    zones = []
    if upper_face_score < 0.55:
        zones.append("Upper Face (Eyes/Glasses/Hat)")
    if lower_face_score < 0.55:
        zones.append("Lower Face (Nose/Mouth/Mask)")
        
    zone_str = ", ".join(zones) if zones else "No Major Local Occlusion"
    return round(float(occlusion_severity), 2), zone_str, mask_grid

class OccludedFaceInferenceEngine:
    """
    Production Inference Engine for OccuPose-BroadDictNet.
    Loads weight checkpoints ONCE and batch-evaluates test images efficiently on GPU/CPU.
    """
    def __init__(self, weights_dir: str = "./weights", gallery_dir: str = "Face_Dataset/name_label"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"========================================================================")
        print(f"--- OccuPose-BroadDictNet Production Inference Engine ---")
        print(f"Device: {self.device} | Weights: {weights_dir}")
        print(f"========================================================================\n")
        
        self.pipeline = OccuPoseBroadDictPipeline(num_enrolled_classes=100, feature_dim=512).to(self.device)
        self.pipeline.load_pretrained_weights(checkpoint_dir=weights_dir, device=self.device)
        
        if os.path.exists(gallery_dir):
            self.pipeline.enroll_dataset(gallery_dir, device=self.device)
            
        self.pipeline.eval()

    def process_single_image(self, image_path: str, output_dir: str = "./inference_results"):
        """
        Runs inference on a single image and saves annotated output image.
        """
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            print(f"❌ Error: Image file invalid or empty: {image_path}")
            return []

        try:
            pil_img, img_tensor, (orig_w, orig_h) = preprocess_image(image_path)
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return []

        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            results = self.pipeline(img_tensor, score_threshold=0.3)

        os.makedirs(output_dir, exist_ok=True)
        draw = ImageDraw.Draw(pil_img)
        
        processed_faces = []
        scale_x = orig_w / 224.0
        scale_y = orig_h / 224.0
        
        with torch.no_grad():
            for idx, res in enumerate(results):
                box = res['box']
                identity = res['identity']
                confidence = res['confidence']
                is_occluded = res['is_occluded']
                attributes = res['attributes']
                
                # Scaled box for original image resolution
                x1_224, y1_224, x2_224, y2_224 = box
                orig_x1 = max(0, min(orig_w, int(x1_224 * scale_x)))
                orig_y1 = max(0, min(orig_h, int(y1_224 * scale_y)))
                orig_x2 = max(0, min(orig_w, int(x2_224 * scale_x)))
                orig_y2 = max(0, min(orig_h, int(y2_224 * scale_y)))
                
                # Crop face patch from 224 tensor for ANet spatial mask breakdown
                crop_patch = img_tensor[:, :, max(0, y1_224):min(224, y2_224), max(0, x1_224):min(224, x2_224)]
                if crop_patch.shape[2] > 5 and crop_patch.shape[3] > 5:
                    resized_crop = F.interpolate(crop_patch, size=(112, 112), mode='bilinear', align_corners=False)
                    attr_logits, occ_mask, _ = self.pipeline.attribute_parser(resized_crop)
                    severity_pct, zone_desc, mask_grid = analyze_occlusion_mask(occ_mask)
                else:
                    severity_pct, zone_desc = 0.0, "Clean"
                    
                face_summary = {
                    'box_224': [x1_224, y1_224, x2_224, y2_224],
                    'box_orig': [orig_x1, orig_y1, orig_x2, orig_y2],
                    'identity': identity,
                    'confidence': confidence,
                    'is_occluded': is_occluded,
                    'severity_pct': severity_pct,
                    'zone_desc': zone_desc,
                    'attributes': attributes
                }
                processed_faces.append(face_summary)
                
                # Draw bounding box & identity tag on visualization image
                box_color = (255, 50, 50) if is_occluded else (50, 255, 50)
                draw.rectangle([orig_x1, orig_y1, orig_x2, orig_y2], outline=box_color, width=max(3, int(orig_w / 100)))
                
                label_text = f"{identity} ({confidence:.0%}) | Occ: {severity_pct:.0f}%"
                text_height = max(18, int(orig_h / 25))
                font_box_y1 = max(0, orig_y1 - text_height)
                draw.rectangle([orig_x1, font_box_y1, orig_x1 + len(label_text) * 8, max(0, orig_y1)], fill=box_color)
                draw.text((orig_x1 + 3, font_box_y1 + 2), label_text, fill=(255, 255, 255))
                
        out_filename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f"predicted_{out_filename}")
        pil_img.save(save_path)
        return processed_faces, save_path

    def predict_directory(self, image_dir: str, output_dir: str = "./inference_results"):
        """
        Iterates over all image files in image_dir, runs inference, and writes a CSV summary report.
        """
        if not os.path.exists(image_dir):
            print(f"❌ Directory not found: {image_dir}")
            return
            
        image_files = []
        for root, _, files in os.walk(image_dir):
            for f in sorted(files):
                if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    image_files.append(os.path.join(root, f))
                    
        print(f"\n========================================================================")
        print(f"--- Processing Batch Directory: {image_dir} ({len(image_files)} images) ---")
        print(f"========================================================================\n")
        
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "test_predictions_summary.csv")
        
        start_time = time.time()
        summary_records = []
        
        for idx, img_path in enumerate(image_files, 1):
            fname = os.path.basename(img_path)
            print(f"[{idx}/{len(image_files)}] Processing: {fname}...")
            
            faces, save_p = self.process_single_image(img_path, output_dir=output_dir)
            
            if len(faces) == 0:
                print(f"   ↳ No face detected.")
                summary_records.append({
                    'image_filename': fname,
                    'num_faces_detected': 0,
                    'face_id': 0,
                    'predicted_identity': 'none',
                    'confidence': '0.00%',
                    'occlusion_flag': False,
                    'occlusion_severity_pct': '0.0%',
                    'affected_zones': 'None',
                    'box_coords': '[]'
                })
            else:
                for f_idx, face in enumerate(faces, 1):
                    print(f"   ↳ Face #{f_idx}: {face['identity'].upper()} | Conf: {face['confidence']:.2%} | Occ: {face['is_occluded']} ({face['severity_pct']}%)")
                    summary_records.append({
                        'image_filename': fname,
                        'num_faces_detected': len(faces),
                        'face_id': f_idx,
                        'predicted_identity': face['identity'],
                        'confidence': f"{face['confidence']:.2%}",
                        'occlusion_flag': face['is_occluded'],
                        'occlusion_severity_pct': f"{face['severity_pct']}%",
                        'affected_zones': face['zone_desc'],
                        'box_coords': str(face['box_orig'])
                    })
                    
        # Write CSV Summary Report
        fieldnames = [
            'image_filename', 'num_faces_detected', 'face_id', 
            'predicted_identity', 'confidence', 'occlusion_flag', 
            'occlusion_severity_pct', 'affected_zones', 'box_coords'
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_records)
            
        elapsed = time.time() - start_time
        print(f"\n========================================================================")
        print(f"✓ Completed inference on {len(image_files)} test images in {elapsed:.2f}s ({elapsed/max(1, len(image_files)):.2f}s/img)")
        print(f"✓ Output annotated images saved to : {output_dir}")
        print(f"✓ CSV summary report saved to      : {csv_path}")
        print(f"========================================================================\n")

def predict_occluded_faces(image_path: str, weights_dir: str = "./weights", output_dir: str = "./inference_results"):
    engine = OccludedFaceInferenceEngine(weights_dir=weights_dir)
    if os.path.isdir(image_path):
        engine.predict_directory(image_path, output_dir=output_dir)
    else:
        engine.process_single_image(image_path, output_dir=output_dir)

def main():
    parser = argparse.ArgumentParser(description="OccuPose-BroadDictNet Occluded Face Prediction Engine")
    parser.add_argument("--image_path", type=str, default="", help="Path to single wild face image file")
    parser.add_argument("--image_dir", type=str, default="", help="Path to directory containing wild face images")
    parser.add_argument("--weights_dir", type=str, default="./weights", help="Path to directory containing trained weight checkpoints")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Directory to save output annotated predictions")
    args = parser.parse_args()
    
    # Default test directory fallback
    test_dir_candidates = [
        "/home/akshat/Documents/Face_Recognition_In_Wild/Test_dataser",
        "./Test_dataser"
    ]
    
    target_dir = args.image_dir
    if not target_dir and not args.image_path:
        for cand in test_dir_candidates:
            if os.path.exists(cand) and os.path.isdir(cand):
                target_dir = cand
                break

    engine = OccludedFaceInferenceEngine(weights_dir=args.weights_dir)

    if args.image_path:
        engine.process_single_image(args.image_path, output_dir=args.output_dir)
    elif target_dir and os.path.exists(target_dir):
        engine.predict_directory(target_dir, output_dir=args.output_dir)
    else:
        print("Usage: python3 predict_occluded_faces.py --image_dir /path/to/Test_dataser --weights_dir ./weights")

if __name__ == "__main__":
    main()
