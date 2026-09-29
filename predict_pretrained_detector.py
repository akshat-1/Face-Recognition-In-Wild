import os
import csv
import math
import time
import torch
import numpy as np
from PIL import Image, ImageDraw

from models.detector import FasterRCNNFaceDetector, SOTAFaceDetector

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def generate_predictions_grid(output_dir: str, grid_filename: str = "all_predictions_grid.jpg", thumb_size=(320, 320)):
    """
    Creates a composite grid figure of all pretrained detector outputs.
    """
    pred_files = []
    if os.path.exists(output_dir):
        for root, _, files in os.walk(output_dir):
            for f in sorted(files):
                if f.startswith("predicted_") and f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    pred_files.append(os.path.join(root, f))
                    
    if len(pred_files) == 0:
        print("No predicted images found for grid compilation.")
        return None

    num_images = len(pred_files)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / float(cols))
    
    tw, th = thumb_size
    header_h = 35
    padding = 10
    banner_h = 70
    
    grid_w = cols * (tw + padding) + padding
    grid_h = rows * (th + header_h + padding) + padding + banner_h
    
    grid_img = Image.new('RGB', (grid_w, grid_h), color=(240, 242, 245))
    draw = ImageDraw.Draw(grid_img)
    
    # Top banner title
    draw.rectangle([0, 0, grid_w, banner_h], fill=(20, 40, 70))
    title_text = f"Pretrained MTCNN + OpenCV YuNet ONNX Face Detection Grid ({num_images} Scenes)"
    draw.text((padding + 10, 22), title_text, fill=(255, 255, 255))
    
    for idx, img_path in enumerate(pred_files):
        r = idx // cols
        c = idx % cols
        
        cell_x = padding + c * (tw + padding)
        cell_y = banner_h + padding + r * (th + header_h + padding)
        
        try:
            pimg = Image.open(img_path).convert('RGB')
            pimg = pimg.resize((tw, th), Image.BILINEAR)
            grid_img.paste(pimg, (cell_x, cell_y + header_h))
            
            fname = os.path.basename(img_path).replace("predicted_", "")
            draw.rectangle([cell_x, cell_y, cell_x + tw, cell_y + header_h], fill=(220, 225, 230))
            draw.text((cell_x + 5, cell_y + 8), f"#{idx+1}: {fname[:25]}", fill=(10, 20, 40))
            
            draw.rectangle([cell_x, cell_y + header_h, cell_x + tw, cell_y + header_h + th], outline=(180, 190, 200), width=2)
        except Exception:
            pass
            
    grid_path = os.path.join(output_dir, grid_filename)
    grid_img.save(grid_path, quality=95)
    print(f"✓ Saved composite grid figure to: {grid_path}")
    return grid_path

def run_pretrained_detector_inference(image_dir: str, weights_dir: str = "./weights", output_dir: str = "./results/pretrained"):
    """
    Runs inference using ONLY the pretrained Faster-RCNN (ResNet-50 FPN) + MTCNN + YuNet detector.
    """
    print(f"========================================================================")
    print(f"--- Pretrained Faster-RCNN (ResNet-50 FPN) + SOTA Face Detector ---")
    print(f"Input Directory: {image_dir} | Output Directory: {output_dir}")
    print(f"========================================================================\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = FasterRCNNFaceDetector(weights_dir=weights_dir).to(device)
    detector.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = []
    for root, _, files in os.walk(image_dir):
        for f in sorted(files):
            if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                image_files.append(os.path.join(root, f))
                
    csv_path = os.path.join(output_dir, "test_predictions_summary.csv")
    summary_records = []
    start_time = time.time()
    
    for idx, img_path in enumerate(image_files, 1):
        fname = os.path.basename(img_path)
        print(f"[{idx}/{len(image_files)}] Processing: {fname}...")
        
        try:
            pil_img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = pil_img.size
            
            # Normalize to [-1, 1] tensor for detector
            np_img = np.array(pil_img).transpose(2, 0, 1)
            img_tensor = (torch.tensor(np_img).unsqueeze(0).float() / 127.5) - 1.0
            img_tensor = img_tensor.to(device)
        except Exception as e:
            print(f"❌ Error reading image {fname}: {e}")
            continue
            
        with torch.no_grad():
            boxes, scores, landmarks = detector(img_tensor, score_threshold=0.4)
            
        draw = ImageDraw.Draw(pil_img)
        num_faces = boxes.size(0)
        
        if num_faces == 0:
            print(f"   ↳ No face detected.")
            summary_records.append({
                'image_filename': fname,
                'num_faces_detected': 0,
                'face_id': 0,
                'detector_score': '0.00%',
                'box_coords': '[]',
                'landmarks_5pt': '[]'
            })
        else:
            for f_idx in range(num_faces):
                box = boxes[f_idx].cpu().numpy().astype(int)
                score = float(scores[f_idx].item())
                lm = landmarks[f_idx].cpu().numpy().astype(int) if landmarks.size(0) > f_idx else np.zeros((5, 2), dtype=int)
                
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(orig_w, box[2]), min(orig_h, box[3])
                
                print(f"   ↳ Face #{f_idx+1}: Box [{x1}, {y1}, {x2}, {y2}] | Detector Score: {score:.2%}")
                
                summary_records.append({
                    'image_filename': fname,
                    'num_faces_detected': num_faces,
                    'face_id': f_idx + 1,
                    'detector_score': f"{score:.2%}",
                    'box_coords': str([x1, y1, x2, y2]),
                    'landmarks_5pt': str(lm.tolist())
                })
                
                # Draw bounding box
                box_color = (0, 220, 255) # Cyan box for pretrained detector
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=max(3, int(orig_w / 120)))
                
                # Draw 5 facial landmark points (eyes, nose, mouth)
                for pt in lm:
                    px, py = pt[0], pt[1]
                    if px > 0 and py > 0:
                        r = max(3, int(orig_w / 150))
                        draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 215, 0), outline=(255, 255, 255))
                        
                # Label tag
                label_text = f"Face #{f_idx+1} | Conf: {score:.0%}"
                text_height = max(18, int(orig_h / 25))
                font_box_y1 = max(0, y1 - text_height)
                draw.rectangle([x1, font_box_y1, x1 + len(label_text) * 8, max(0, y1)], fill=box_color)
                draw.text((x1 + 3, font_box_y1 + 2), label_text, fill=(0, 0, 0))
                
        out_filename = f"predicted_{fname}"
        save_path = os.path.join(output_dir, out_filename)
        pil_img.save(save_path, quality=95)
        
    # Save CSV Summary
    fieldnames = ['image_filename', 'num_faces_detected', 'face_id', 'detector_score', 'box_coords', 'landmarks_5pt']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_records)
        
    grid_path = generate_predictions_grid(output_dir, grid_filename="all_predictions_grid.jpg")
    
    elapsed = time.time() - start_time
    print(f"\n========================================================================")
    print(f"✓ Completed pretrained detector inference in {elapsed:.2f}s ({elapsed/max(1, len(image_files)):.2f}s/img)")
    print(f"✓ Output annotated images saved to : {output_dir}")
    print(f"✓ CSV summary report saved to      : {csv_path}")
    if grid_path:
        print(f"✓ Composite grid figure saved to   : {grid_path}")
    print(f"========================================================================\n")

if __name__ == "__main__":
    run_pretrained_detector_inference(
        image_dir="/home/akshat/Documents/Face_Recognition_In_Wild/Test_dataser",
        weights_dir="/home/akshat/Documents/Face_Recognition_In_Wild/weights",
        output_dir="/home/akshat/Documents/Face_Recognition_In_Wild/results/pretrained"
    )
