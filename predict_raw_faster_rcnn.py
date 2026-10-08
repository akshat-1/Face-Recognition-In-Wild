import os
import csv
import math
import time
import torch
import numpy as np
import torchvision.models.detection as detection
from PIL import Image, ImageDraw

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def generate_predictions_grid(output_dir: str, grid_filename: str = "all_predictions_grid.jpg", thumb_size=(320, 320)):
    """
    Creates a composite grid figure of all raw pretrained Faster-RCNN outputs.
    """
    pred_files = []
    if os.path.exists(output_dir):
        for root, _, files in os.walk(output_dir):
            for f in sorted(files):
                if (f.startswith("raw_prediction_") or f.startswith("predicted_")) and f.lower().endswith(VALID_IMAGE_EXTENSIONS):
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
    draw.rectangle([0, 0, grid_w, banner_h], fill=(40, 30, 70))
    title_text = f"Raw Pretrained Faster-RCNN (ResNet-50 FPN) Bounding Box Grid ({num_images} Scenes)"
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
            
            fname_clean = os.path.basename(img_path)
            for pfx in ["raw_prediction_", "predicted_"]:
                if fname_clean.startswith(pfx):
                    fname_clean = fname_clean[len(pfx):]
            draw.rectangle([cell_x, cell_y, cell_x + tw, cell_y + header_h], fill=(220, 225, 230))
            draw.text((cell_x + 5, cell_y + 8), f"#{idx+1}: {fname_clean[:25]}", fill=(10, 20, 40))
            
            draw.rectangle([cell_x, cell_y + header_h, cell_x + tw, cell_y + header_h + th], outline=(180, 190, 200), width=2)
        except Exception:
            pass
            
    grid_path = os.path.join(output_dir, grid_filename)
    grid_img.save(grid_path, quality=95)
    print(f"✓ Saved composite grid figure to: {grid_path}")
    return grid_path

def run_raw_faster_rcnn_inference(target_path: str, output_dir: str = "./results/raw_pretrained"):
    """
    Runs inference using ONLY the raw pretrained Faster-RCNN (ResNet-50 FPN COCO) model.
    """
    print(f"========================================================================")
    print(f"--- Raw Pretrained Faster-RCNN (ResNet-50 FPN) Inference ---")
    print(f"Target Path: {target_path} | Output Directory: {output_dir}")
    print(f"========================================================================\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = detection.fasterrcnn_resnet50_fpn(weights=detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = []
    if os.path.isfile(target_path):
        image_files = [target_path]
    elif os.path.isdir(target_path):
        for root, _, files in os.walk(target_path):
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
            
            # Normalize to [0, 1] tensor for torchvision Faster-RCNN
            np_img = np.array(pil_img).transpose(2, 0, 1)
            img_tensor = torch.tensor(np_img).float() / 255.0
            img_tensor = img_tensor.to(device)
        except Exception as e:
            print(f"❌ Error reading image {fname}: {e}")
            continue
            
        with torch.no_grad():
            preds = model([img_tensor])[0]
            boxes = preds['boxes']
            scores = preds['scores']
            labels = preds['labels']
            
            # Filter person / bounding box proposals (label == 1, score >= 0.3)
            mask = (labels == 1) & (scores >= 0.3)
            p_boxes = boxes[mask]
            p_scores = scores[mask]
            
            # Fallback to top-scoring box if score < 0.3
            if p_boxes.size(0) == 0 and boxes.size(0) > 0:
                p_boxes = boxes[:1]
                p_scores = scores[:1]
                
        draw = ImageDraw.Draw(pil_img)
        num_boxes = p_boxes.size(0)
        
        if num_boxes == 0:
            print(f"   ↳ No bounding box detected.")
            summary_records.append({
                'image_filename': fname,
                'num_boxes_detected': 0,
                'box_id': 0,
                'faster_rcnn_score': '0.00%',
                'box_coords': '[]'
            })
        else:
            for b_idx in range(num_boxes):
                box = p_boxes[b_idx].cpu().numpy().astype(int)
                score = float(p_scores[b_idx].item())
                
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(orig_w, box[2]), min(orig_h, box[3])
                
                print(f"   ↳ BBox #{b_idx+1}: [{x1}, {y1}, {x2}, {y2}] | Score: {score:.2%}")
                
                summary_records.append({
                    'image_filename': fname,
                    'num_boxes_detected': num_boxes,
                    'box_id': b_idx + 1,
                    'faster_rcnn_score': f"{score:.2%}",
                    'box_coords': str([x1, y1, x2, y2])
                })
                
                # Draw raw Faster-RCNN bounding box in Magenta
                box_color = (255, 0, 128)
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=max(3, int(orig_w / 120)))
                
                # Label tag
                label_text = f"Raw Faster-RCNN | {score:.0%}"
                text_height = max(18, int(orig_h / 25))
                font_box_y1 = max(0, y1 - text_height)
                draw.rectangle([x1, font_box_y1, x1 + len(label_text) * 8, max(0, y1)], fill=box_color)
                draw.text((x1 + 3, font_box_y1 + 2), label_text, fill=(255, 255, 255))
                
        clean_fname = fname
        for pfx in ["raw_prediction_", "predicted_"]:
            if clean_fname.startswith(pfx):
                clean_fname = clean_fname[len(pfx):]
                
        out_filename = f"raw_prediction_{clean_fname}"
        save_path = os.path.join(output_dir, out_filename)
        pil_img.save(save_path, quality=95)
        
    # Save CSV Summary
    fieldnames = ['image_filename', 'num_boxes_detected', 'box_id', 'faster_rcnn_score', 'box_coords']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_records)
        
    grid_path = generate_predictions_grid(output_dir, grid_filename="all_predictions_grid.jpg")
    
    elapsed = time.time() - start_time
    print(f"\n========================================================================")
    print(f"✓ Completed raw Faster-RCNN inference in {elapsed:.2f}s ({elapsed/max(1, len(image_files)):.2f}s/img)")
    print(f"✓ Output annotated images saved to : {output_dir}")
    print(f"✓ CSV summary report saved to      : {csv_path}")
    if grid_path:
        print(f"✓ Composite grid figure saved to   : {grid_path}")
    print(f"========================================================================\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Raw Pretrained Faster-RCNN Base Detector Inference")
    parser.add_argument("--image_path", type=str, default="", help="Path to single image file")
    parser.add_argument("--image_dir", type=str, default="", help="Path to directory containing images")
    parser.add_argument("--output_dir", type=str, default="./results/raw_pretrained", help="Output directory")
    args = parser.parse_args()
    
    target_path = args.image_path if args.image_path else (args.image_dir if args.image_dir else "./Test_dataser")
    run_raw_faster_rcnn_inference(target_path=target_path, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
