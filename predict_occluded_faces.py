import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from pipeline.wild_face_pipeline import OccuPoseBroadDictPipeline
from models.anet_attribute import ANetAttributeParser
from dataset import get_default_transform

def preprocess_image(image_path: str, image_size=(224, 224)):
    """
    Loads and preprocesses an image file into a normalized tensor [-1, 1].
    """
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    
    # Standard transform for full scene input
    transform = get_default_transform(image_size=image_size, is_train=False)
    img_tensor = transform(img).unsqueeze(0) # (1, 3, H, W)
    return img, img_tensor, (orig_w, orig_h)

def analyze_occlusion_mask(occ_mask: torch.Tensor):
    """
    Analyzes the 7x7 spatial occlusion map M_spatial returned by ANet.
    Returns occlusion severity percentage and spatial zone breakdown.
    """
    # occ_mask is (1, 1, 7, 7) or (1, 7, 7)
    mask_grid = occ_mask.squeeze().cpu().numpy() # (7, 7) array with values in [0, 1]
    
    # Clean regions have weight ~ 1.0, occluded regions have weight < 0.5
    occlusion_severity = (1.0 - np.mean(mask_grid)) * 100.0
    
    upper_face_score = np.mean(mask_grid[:3, :]) # Top 3 rows (Eyes/Hat/Sunglasses)
    lower_face_score = np.mean(mask_grid[3:, :]) # Bottom 4 rows (Nose/Mouth/Mask)
    
    zones = []
    if upper_face_score < 0.55:
        zones.append("Upper Face (Eyes/Glasses/Hat)")
    if lower_face_score < 0.55:
        zones.append("Lower Face (Nose/Mouth/Mask)")
        
    zone_str = ", ".join(zones) if zones else "No Major Local Occlusion"
    return round(float(occlusion_severity), 2), zone_str, mask_grid

def predict_occluded_faces(image_path: str, weights_dir: str = "./weights", output_dir: str = "./inference_results"):
    """
    Predicts occluded faces in a wild image using trained OccuPose-BroadDictNet weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"========================================================================")
    print(f"--- OccuPose-BroadDictNet Occluded Face Prediction Engine ---")
    print(f"Device: {device} | Weights: {weights_dir}")
    print(f"========================================================================\n")
    
    # 1. Initialize Pipeline & Load Pretrained Checkpoints
    pipeline = OccuPoseBroadDictPipeline(num_enrolled_classes=100, feature_dim=512).to(device)
    pipeline.load_pretrained_weights(checkpoint_dir=weights_dir, device=device)
    pipeline.eval()
    
    # 2. Preprocess Input Wild Image
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return
        
    pil_img, img_tensor, (orig_w, orig_h) = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # 3. Execute End-to-End Occlusion & Recognition Inference
    print(f"\nProcessing wild image: {image_path} ({orig_w}x{orig_h})...")
    with torch.no_grad():
        results = pipeline(img_tensor, score_threshold=0.3)
        
    print(f"\n✓ Found {len(results)} face(s) in scene:")
    
    draw = ImageDraw.Draw(pil_img)
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, res in enumerate(results):
        box = res['box']
        identity = res['identity']
        confidence = res['confidence']
        is_occluded = res['is_occluded']
        attributes = res['attributes']
        
        # Crop face patch for ANet spatial mask breakdown
        x1, y1, x2, y2 = box
        crop_patch = img_tensor[:, :, max(0, y1):min(orig_h, y2), max(0, x1):min(orig_w, x2)]
        if crop_patch.shape[2] > 5 and crop_patch.shape[3] > 5:
            resized_crop = F.interpolate(crop_patch, size=(112, 112), mode='bilinear', align_corners=False)
            attr_logits, occ_mask, _ = pipeline.attribute_parser(resized_crop)
            severity_pct, zone_desc, mask_grid = analyze_occlusion_mask(occ_mask)
        else:
            severity_pct, zone_desc = 0.0, "Clean"
            
        print(f"\n------------------------------------------------------------------------")
        print(f"Face Candidate #{idx + 1}:")
        print(f"  • Bounding Box Coordinates : [{x1}, {y1}, {x2}, {y2}]")
        print(f"  • Predicted Identity       : {identity.upper()} (Confidence: {confidence:.2%})")
        print(f"  • Major Occlusion Flag     : {is_occluded} (Severity: {severity_pct}%)")
        print(f"  • Affected Spatial Zones   : {zone_desc}")
        print(f"  • Top Predicted Attributes :")
        for attr_name, active in list(attributes.items())[:6]:
            flag_str = "YES" if active else "NO"
            print(f"      - {attr_name:25s}: {flag_str}")
            
        # Draw bounding box & identity tag on visualization image
        box_color = (255, 50, 50) if is_occluded else (50, 255, 50)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        
        label_text = f"{identity} ({confidence:.0%}) | Occ: {severity_pct:.0f}%"
        draw.rectangle([x1, max(0, y1 - 20), x1 + len(label_text) * 7, max(0, y1)], fill=box_color)
        draw.text((x1 + 3, max(0, y1 - 18)), label_text, fill=(255, 255, 255))
        
    out_filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"predicted_{out_filename}")
    pil_img.save(save_path)
    print(f"\n========================================================================")
    print(f"✓ Saved annotated prediction visualization to: {save_path}")
    print(f"========================================================================\n")

def main():
    parser = argparse.ArgumentParser(description="OccuPose-BroadDictNet Occluded Face Prediction & Attribute Parsing Engine")
    parser.add_argument("--image_path", type=str, default="", help="Path to single wild face image")
    parser.add_argument("--image_dir", type=str, default="", help="Path to directory containing wild face images")
    parser.add_argument("--weights_dir", type=str, default="./weights", help="Path to directory containing trained weight checkpoints")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Directory to save output annotated predictions")
    args = parser.parse_args()
    
    if args.image_path:
        predict_occluded_faces(args.image_path, weights_dir=args.weights_dir, output_dir=args.output_dir)
    elif args.image_dir and os.path.exists(args.image_dir):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for root, _, files in os.walk(args.image_dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    img_p = os.path.join(root, f)
                    predict_occluded_faces(img_p, weights_dir=args.weights_dir, output_dir=args.output_dir)
    else:
        # Default demo prediction on sample dataset image
        sample_candidates = [
            "Face_Dataset/name_label/ROF/masked/alexander_zverev_wearing_mask/000001.jpg",
            "Face_Dataset/name_label/ROF/masked/amy_klobuchar_wearing_mask/000001.jpg",
            "Face_Dataset/bb_label/darknet/mask-dataset/images/000001.jpg"
        ]
        test_path = None
        for cand in sample_candidates:
            if os.path.exists(cand):
                test_path = cand
                break
                
        if test_path:
            predict_occluded_faces(test_path, weights_dir=args.weights_dir, output_dir=args.output_dir)
        else:
            print("Usage: python3 predict_occluded_faces.py --image_path <path/to/image.jpg> --weights_dir ./weights")

if __name__ == "__main__":
    main()
