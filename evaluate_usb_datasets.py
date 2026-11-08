import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

from inference import load_pipeline, resolve_device
from dataset import canonical_identity_name, pil_to_float_tensor
from models.detector import FasterRCNNFaceDetector


def test_name_recognition(pipeline, face_data_root, output_dir):
    """
    Test name recognition accuracy on face_data/LFW_without_Mask and Masked_LFW_Dataset.
    """
    from inference import predict_image
    print(f"Testing name recognition on {face_data_root}")

    device = next(pipeline.parameters()).device
    total = 0
    correct = 0

    datasets = [
        (face_data_root / "LFW_without_Mask", "dir"),
        (face_data_root / "Masked_LFW_Dataset", "filename")
    ]

    for root, mode in datasets:
        if not root.is_dir():
            print(f"Skipping {root}: directory not found")
            continue

        print(f"Processing {root} (mode: {mode})...")
        images = sorted([p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])

        for i, img_path in enumerate(images):
            if mode == "dir":
                gt_name = canonical_identity_name(img_path.parent.name)
            else:
                gt_name = canonical_identity_name(img_path.stem)

            result = predict_image(pipeline, img_path, device, score_threshold=0.3)
            preds = result.get("predictions", [])
            if preds:
                best_pred = preds[0]
                pred_name = canonical_identity_name(str(best_pred.get("identity", "unknown")))
                if pred_name == gt_name:
                    correct += 1

            total += 1
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(images)} images. Current accuracy: {correct/max(1, total):.2%}")

    accuracy = correct / max(1, total)
    return {"accuracy": accuracy, "total": total, "correct": correct}


def test_bounding_box_accuracy(detector, facemask_root, output_dir):
    """
    Test bounding box accuracy on facemaskyolo.
    facemaskyolo has images/test and labels/test.
    """
    from evaluate_all import box_iou
    from dataset import pil_to_float_tensor
    import torch

    print(f"Testing bounding box accuracy on {facemask_root}")
    image_dir = facemask_root / "images" / "test"
    label_dir = facemask_root / "labels" / "test"

    if not image_dir.is_dir() or not label_dir.is_dir():
        print(f"Skipping bounding box test: {image_dir} or {label_dir} not found")
        return None

    image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    device = next(detector.parameters()).device

    iou_threshold = 0.5
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i, img_path in enumerate(image_paths):
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        gt_boxes = []
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                rgb_img = img.convert("RGB")
        except (OSError, ValueError) as e:
            print(f"Skipping corrupt image {img_path}: {e}")
            continue

        content = label_path.read_text().strip()
        if content:
            for line in content.splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw/2) * w
                    y1 = (yc - bh/2) * h
                    x2 = (xc + bw/2) * w
                    y2 = (yc + bh/2) * h
                    gt_boxes.append([x1, y1, x2, y2])

        tensor = pil_to_float_tensor(rgb_img).mul(2.0).sub(1.0).unsqueeze(0).to(device)
        with torch.inference_mode():
            pred_boxes, pred_scores, _ = detector(tensor, score_threshold=0.3)

        pred_boxes = pred_boxes.detach().cpu().tolist()

        matched_gt = [False] * len(gt_boxes)
        for pb in pred_boxes:
            best_iou = 0
            best_idx = -1
            for idx, gb in enumerate(gt_boxes):
                if matched_gt[idx]:
                    continue
                current_iou = box_iou(pb, gb)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_idx = idx

            if best_iou >= iou_threshold:
                true_positives += 1
                matched_gt[best_idx] = True
            else:
                false_positives += 1

        false_negatives += sum(1 for m in matched_gt if not m)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(image_paths)} images...")

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on USB test datasets")
    parser.add_argument("--face_data", default="/run/media/akshat/Akshat_USB/face_data")
    parser.add_argument("--facemask_yolo", default="/run/media/akshat/Akshat_USB/facemaskyolo")
    parser.add_argument("--weights_dir", default="./weights")
    parser.add_argument("--gallery_dir", default="./Face_Dataset/name_label")
    parser.add_argument("--output_dir", default="./usb_evaluation_results")
    args = parser.parse_args()

    device = resolve_device("auto")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline (includes detector)
    pipeline = load_pipeline(args.weights_dir, args.gallery_dir, device)

    # Test Name Recognition
    name_rec_results = test_name_recognition(pipeline, Path(args.face_data), output_dir)

    # Test Bounding Box
    bbox_results = test_bounding_box_accuracy(pipeline.detector, Path(args.facemask_yolo), output_dir)

    # Generate Report
    report = {
        "device": str(device),
        "weights_dir": str(args.weights_dir),
        "gallery_dir": str(args.gallery_dir),
        "name_recognition": name_rec_results,
        "bounding_box": bbox_results
    }

    report_path = output_dir / "test_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nFinal Test Report:")
    print(f"------------------")
    print(f"Name Recognition Accuracy: {name_rec_results['accuracy']:.2%}")
    print(f"  Total images: {name_rec_results['total']}")
    print(f"  Correct: {name_rec_results['correct']}")

    if bbox_results:
        print(f"Bounding Box Metrics (IoU >= 0.5):")
        print(f"  Precision: {bbox_results['precision']:.2%}")
        print(f"  Recall:    {bbox_results['recall']:.2%}")
        print(f"  F1-Score:  {bbox_results['f1']:.2%}")
        print(f"  TP: {bbox_results['true_positives']}, FP: {bbox_results['false_positives']}, FN: {bbox_results['false_negatives']}")

    print(f"\nDetailed report saved to {report_path}")


if __name__ == "__main__":
    main()
