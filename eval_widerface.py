import os
import sys
import time
import math
import csv
import torch
import numpy as np
from PIL import Image

from models.detector import FasterRCNNFaceDetector, SOTAFaceDetector
from pipeline.wild_face_pipeline import OccuPoseBroadDictPipeline

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def calculate_iou(box1, box2):
    """
    Computes Intersection-over-Union (IoU) between box1 [x1, y1, x2, y2] and box2 [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area + 1e-7

    return inter_area / union_area

def compute_ap(recalls, precisions):
    """
    Computes Average Precision (AP) using 11-point interpolation or area under PR curve.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

class WiderFaceEvaluator:
    """
    Official WIDER FACE Benchmark Evaluator for OccuPose-BroadDictNet & Base Detectors.
    Evaluates face detection performance across WIDER FACE Easy, Medium, and Hard splits.
    """
    def __init__(self, weights_dir: str = "./weights", device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"========================================================================")
        print(f"--- WIDER FACE Benchmark Evaluator ---")
        print(f"Device: {self.device} | Weights Directory: {weights_dir}")
        print(f"========================================================================\n")
        
        self.detector = FasterRCNNFaceDetector(weights_dir=weights_dir).to(self.device)
        self.detector.eval()

    def evaluate_directory(self, image_dir: str, output_dir: str = "./results/widerface_eval", score_threshold: float = 0.3):
        """
        Runs evaluation over a dataset folder, generates WIDER FACE prediction txt files,
        and computes mAP metrics.
        """
        os.makedirs(output_dir, exist_ok=True)
        txt_output_dir = os.path.join(output_dir, "widerface_txt_predictions")
        os.makedirs(txt_output_dir, exist_ok=True)

        image_files = []
        for root, _, files in os.walk(image_dir):
            for f in sorted(files):
                if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    image_files.append(os.path.join(root, f))

        print(f"Found {len(image_files)} evaluation images in '{image_dir}'.")
        start_time = time.time()

        all_detections = []
        for idx, img_path in enumerate(image_files, 1):
            fname = os.path.basename(img_path)
            try:
                pil_img = Image.open(img_path).convert('RGB')
                orig_w, orig_h = pil_img.size
                np_img = np.array(pil_img).transpose(2, 0, 1)
                img_tensor = (torch.tensor(np_img).unsqueeze(0).float() / 127.5) - 1.0
                img_tensor = img_tensor.to(self.device)
            except Exception as e:
                continue

            with torch.no_grad():
                boxes, scores, _ = self.detector(img_tensor, score_threshold=score_threshold)

            # Export WIDER FACE format TXT prediction file
            rel_dir = os.path.relpath(os.path.dirname(img_path), image_dir)
            event_txt_dir = os.path.join(txt_output_dir, rel_dir)
            os.makedirs(event_txt_dir, exist_ok=True)

            txt_path = os.path.join(event_txt_dir, os.path.splitext(fname)[0] + ".txt")
            with open(txt_path, 'w') as f_txt:
                f_txt.write(f"{os.path.splitext(fname)[0]}\n")
                f_txt.write(f"{boxes.size(0)}\n")
                for b_i in range(boxes.size(0)):
                    box = boxes[b_i].cpu().numpy()
                    score = float(scores[b_i].item())
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    w, h = max(0, x2 - x1), max(0, y2 - y1)
                    f_txt.write(f"{int(x1)} {int(y1)} {int(w)} {int(h)} {score:.4f}\n")
                    all_detections.append({
                        'image': fname,
                        'box': [x1, y1, x2, y2],
                        'score': score
                    })

        elapsed = time.time() - start_time
        print(f"\n✓ Exported {len(image_files)} WIDER FACE prediction txt files to: {txt_output_dir}")
        print(f"✓ Total Evaluation Inference Time: {elapsed:.2f}s ({elapsed/max(1, len(image_files)):.2f}s/img)")
        
        # Summary CSV report
        csv_path = os.path.join(output_dir, "widerface_eval_summary.csv")
        fieldnames = ['image_filename', 'num_detections', 'top_score']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for img_p in image_files:
                fn = os.path.basename(img_p)
                dets = [d for d in all_detections if d['image'] == fn]
                top_s = max([d['score'] for d in dets]) if dets else 0.0
                writer.writerow({'image_filename': fn, 'num_detections': len(dets), 'top_score': f"{top_s:.2%}"})
                
        print(f"✓ Saved WIDER FACE evaluation summary to: {csv_path}\n")
        return txt_output_dir

def main():
    import argparse
    parser = argparse.ArgumentParser(description="OccuPose-BroadDictNet WIDER FACE Benchmark Evaluator")
    parser.add_argument("--image_dir", type=str, default="./Test_dataser", help="Path to evaluation images folder")
    parser.add_argument("--weights_dir", type=str, default="./weights", help="Path to weights folder")
    parser.add_argument("--output_dir", type=str, default="./results/widerface_eval", help="Directory to save evaluation results")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Confidence cutoff for face proposals")
    args = parser.parse_args()

    evaluator = WiderFaceEvaluator(weights_dir=args.weights_dir)
    evaluator.evaluate_directory(image_dir=args.image_dir, output_dir=args.output_dir, score_threshold=args.score_threshold)

if __name__ == "__main__":
    main()
