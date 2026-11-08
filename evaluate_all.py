"""Run one reproducible evaluation over every dataset in ``Evaluation/``.

The supplied benchmark data contains face bounding-box annotations, not person
identity annotations.  This runner therefore reports detection metrics
(precision, recall, F1, AP, and per-image coverage) and, when full pipeline
inference is enabled, reports recognition operating statistics such as known
versus rejected predictions.  It never presents those operating statistics as
identity accuracy.

Example::

    python evaluate_all.py \
        --data_root Evaluation \
        --weights_dir weights \
        --gallery_dir Face_Dataset/name_label \
        --output_dir evaluation_results \
        --device cuda \
        --mode full

The command writes JSON, CSV, Markdown, and per-dataset JSONL prediction
artifacts.  Use ``--mode detector`` when only localization metrics are needed.
"""

from __future__ import annotations
import torch
import torchvision
from PIL import Image
from pathlib import Path
import json
import time
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence
import argparse
import csv
import re

import argparse
import csv
import json
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence



IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IOU_THRESHOLD = 0.50
AFW_BOX_RE = re.compile(
    r"\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*;\s*"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]"
)


DATASETS: Mapping[str, Dict[str, Any]] = OrderedDict(
    (
        (
            "widerface",
            {
                "description": "WIDER FACE validation",
                "image_path": "WIDER_val/images",
                "gt_type": "wider",
                "gt_path": "wider_face_split/wider_face_val_bbx_gt.txt",
            },
        ),
        (
            "ufdd",
            {
                "description": "Unconstrained Face Detection Dataset validation",
                "image_path": "UFDD-V1-05042018/UFDD_val/images",
                "gt_type": "ufdd",
                "gt_path": "UFDD-V1-05042018/UFDD-annotationfile/UFDD_split/UFDD_val_bbx_gt.txt",
            },
        ),
        (
            "afw",
            {
                "description": "Annotated Faces in the Wild",
                "image_path": "AFWDataset-master/images",
                "gt_type": "afw",
                "gt_path": "AFWDataset-master",
            },
        ),
        (
            "darkface",
            {
                "description": "DARKFACE low-light test sample",
                "image_path": "track1.2_test_sample-DARKFACE",
                "gt_type": None,
                "gt_path": None,
            },
        ),
    )
)


def _box_from_xywh(parts: Sequence[str]) -> List[float]:
    x, y, width, height = (float(parts[index]) for index in range(4))
    return [x, y, x + width, y + height]


def load_ufdd_gt(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load UFDD's image/count/box text format."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result: Dict[str, List[Dict[str, Any]]] = {}
    index = 0
    while index < len(lines):
        image_key = Path(lines[index]).as_posix()
        index += 1
        if index >= len(lines) or not lines[index].isdigit():
            raise ValueError(f"Malformed UFDD annotation near {image_key!r}")
        box_count = int(lines[index])
        index += 1
        boxes = []
        for _ in range(box_count):
            if index >= len(lines):
                raise ValueError(f"Truncated UFDD annotation for {image_key!r}")
            parts = lines[index].split()
            if len(parts) < 4:
                raise ValueError(f"Malformed UFDD box for {image_key!r}: {lines[index]!r}")
            box = _box_from_xywh(parts)
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append({"box": box, "height": box[3] - box[1]})
            index += 1
        result[image_key] = boxes
    return result


def load_afw_gt(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Parse the AFW MATLAB-exported ``filenames.txt`` and ``faces.txt``."""
    filenames = [item for item in (root / "filenames.txt").read_text(encoding="utf-8").split("*") if item]
    face_chunks = (root / "faces.txt").read_text(encoding="utf-8").split("*")
    if len(filenames) != len([item for item in face_chunks if item]):
        raise ValueError(
            "AFW filenames.txt and faces.txt contain different numbers of image records"
        )

    result: Dict[str, List[Dict[str, Any]]] = {}
    for filename, chunk in zip(filenames, face_chunks):
        boxes = []
        for match in AFW_BOX_RE.finditer(chunk):
            x1, y1, x2, y2 = (float(match.group(index)) for index in range(1, 5))
            if x2 > x1 and y2 > y1:
                boxes.append({"box": [x1, y1, x2, y2], "height": y2 - y1})
        result[Path(filename).as_posix()] = boxes
    return result


def load_ground_truth(dataset_name: str, spec: Mapping[str, Any], data_root: Path):
    gt_type = spec.get("gt_type")
    if gt_type is None:
        return None
    gt_path = data_root / str(spec["gt_path"])
    if gt_type == "wider":
        from eval_widerface import load_widerface_gt, normalize_widerface_image_key

        parsed = load_widerface_gt(str(gt_path))
        return {
            normalize_widerface_image_key(key): [
                {"box": list(item["box"]), "height": float(item["h"])}
                for item in items
            ]
            for key, items in parsed.items()
        }
    if gt_type == "ufdd":
        return load_ufdd_gt(gt_path)
    if gt_type == "afw":
        return load_afw_gt(gt_path)
    raise ValueError(f"Unsupported ground-truth type for {dataset_name}: {gt_type}")


def image_key(path: Path, image_root: Path) -> str:
    return path.relative_to(image_root).as_posix()


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    return intersection / max(first_area + second_area - intersection, 1e-12)


def _average_precision(records: Sequence[Dict[str, Any]], ground_truth: Mapping[str, List[Dict[str, Any]]], height_filter=None) -> float:
    filtered_gt = {}
    for key, items in ground_truth.items():
        filtered_gt[key] = [
            item for item in items
            if height_filter is None or height_filter(float(item["height"]))
        ]
    total_gt = sum(len(items) for items in filtered_gt.values())
    if total_gt == 0:
        return 0.0

    matched = {key: [False] * len(items) for key, items in filtered_gt.items()}
    true_positives = []
    false_positives = []
    for record in sorted(records, key=lambda item: float(item["score"]), reverse=True):
        key = record["image"]
        best_iou = IOU_THRESHOLD
        best_index = -1
        for index, item in enumerate(filtered_gt.get(key, [])):
            if matched[key][index]:
                continue
            current_iou = box_iou(record["box"], item["box"])
            if current_iou >= best_iou:
                best_iou = current_iou
                best_index = index
        if best_index >= 0:
            matched[key][best_index] = True
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    cumulative_tp = []
    cumulative_fp = []
    running_tp = running_fp = 0
    for true_positive, false_positive in zip(true_positives, false_positives):
        running_tp += true_positive
        running_fp += false_positive
        cumulative_tp.append(running_tp)
        cumulative_fp.append(running_fp)
    recalls = [value / total_gt for value in cumulative_tp]
    precisions = [
        tp / max(tp + fp, 1)
        for tp, fp in zip(cumulative_tp, cumulative_fp)
    ]
    recall_points = [0.0] + recalls + [1.0]
    precision_points = [0.0] + precisions + [0.0]
    for index in range(len(precision_points) - 1, 0, -1):
        precision_points[index - 1] = max(precision_points[index - 1], precision_points[index])
    return float(
        sum(
            (recall_points[index + 1] - recall_points[index]) * precision_points[index + 1]
            for index in range(len(recall_points) - 1)
            if recall_points[index + 1] != recall_points[index]
        )
    )


def detection_metrics(
    image_keys: Sequence[str],
    ground_truth: Mapping[str, List[Dict[str, Any]]],
    prediction_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate image coverage, box P/R/F1, AP, and matched IoU."""
    predictions_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for record in prediction_records:
        predictions_by_image.setdefault(record["image"], []).append(record)

    true_positive = false_positive = false_negative = 0
    matched_ious = []
    images_with_detection = 0
    for key in image_keys:
        predictions = sorted(predictions_by_image.get(key, []), key=lambda item: item["score"], reverse=True)
        if predictions:
            images_with_detection += 1
        targets = ground_truth.get(key, [])
        matched = [False] * len(targets)
        for prediction in predictions:
            best_iou = IOU_THRESHOLD
            best_index = -1
            for index, target in enumerate(targets):
                if matched[index]:
                    continue
                current_iou = box_iou(prediction["box"], target["box"])
                if current_iou >= best_iou:
                    best_iou = current_iou
                    best_index = index
            if best_index >= 0:
                matched[best_index] = True
                true_positive += 1
                matched_ious.append(best_iou)
            else:
                false_positive += 1
        false_negative += sum(not value for value in matched)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    metrics: Dict[str, Any] = {
        "images": len(image_keys),
        "ground_truth_faces": sum(len(ground_truth.get(key, [])) for key in image_keys),
        "predicted_faces": len(prediction_records),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap50": _average_precision(prediction_records, ground_truth),
        "images_with_detection": images_with_detection,
        "image_detection_rate": images_with_detection / max(len(image_keys), 1),
        "mean_detections_per_image": len(prediction_records) / max(len(image_keys), 1),
        "mean_matched_iou": sum(matched_ious) / max(len(matched_ious), 1),
        "iou_threshold": IOU_THRESHOLD,
    }
    if any(item.get("height") is not None for items in ground_truth.values() for item in items):
        metrics["easy_ap50"] = _average_precision(
            prediction_records, ground_truth, lambda height: height >= 50
        )
        metrics["medium_ap50"] = _average_precision(
            prediction_records, ground_truth, lambda height: 30 <= height < 50
        )
        metrics["hard_ap50"] = _average_precision(
            prediction_records, ground_truth, lambda height: height < 30
        )
    return metrics


def prediction_metrics(predictions_by_image: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    predictions = [item for items in predictions_by_image.values() for item in items]
    known = [item for item in predictions if str(item.get("identity", "unknown")) != "unknown"]
    unknown = [item for item in predictions if str(item.get("identity", "unknown")) == "unknown"]
    occluded = [item for item in predictions if bool(item.get("is_occluded", False))]
    return {
        "predictions": len(predictions),
        "known_identity_predictions": len(known),
        "unknown_predictions": len(unknown),
        "occluded_predictions": len(occluded),
        "known_identity_rate": len(known) / max(len(predictions), 1),
        "unknown_rejection_rate": len(unknown) / max(len(predictions), 1),
        "occlusion_rate": len(occluded) / max(len(predictions), 1),
        "mean_detector_score": sum(float(item.get("detector_score", 0.0)) for item in predictions) / max(len(predictions), 1),
        "mean_recognition_score": sum(float(item.get("recognition_score", 0.0)) for item in predictions) / max(len(predictions), 1),
    }


def run_dataset(
    dataset_name: str,
    spec: Mapping[str, Any],
    data_root: Path,
    output_dir: Path,
    device: torch.device,
    mode: str,
    score_threshold: float,
    detector,
    pipeline,
) -> Dict[str, Any]:
    image_root = data_root / str(spec["image_path"])
    
    paths = image_paths(None, str(image_root))
    keys = [image_key(path, image_root) for path in paths]
    ground_truth = load_ground_truth(dataset_name, spec, data_root)
    missing_gt = []
    if ground_truth is not None:
        missing_gt = sorted(set(keys) - set(ground_truth))
        if missing_gt:
            raise ValueError(
                f"{dataset_name}: ground truth does not cover {len(missing_gt)} images; "
                f"first missing key: {missing_gt[0]}"
            )

    predictions_by_image: Dict[str, List[Dict[str, Any]]] = {}
    prediction_records: List[Dict[str, Any]] = []
    prediction_path = output_dir / "predictions" / f"{dataset_name}.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with prediction_path.open("w", encoding="utf-8") as stream:
        for index, path in enumerate(paths, start=1):
            if mode == "full":
                result = predict_image(pipeline, path, device, score_threshold)
                predictions = result["predictions"]
            else:
                from dataset import pil_to_float_tensor

                with Image.open(path) as source:
                    tensor = pil_to_float_tensor(source.convert("RGB")).mul(2.0).sub(1.0).unsqueeze(0).to(device)
                with torch.inference_mode():
                    boxes, scores, _ = detector(tensor, score_threshold=score_threshold)
                predictions = []
                for box, score in zip(boxes.detach().cpu().tolist(), scores.detach().cpu().tolist()):
                    predictions.append({
                        "box": [float(value) for value in box],
                        "box_xyxy": [float(value) for value in box],
                        "detector_score": float(score),
                    })

            key = keys[index - 1]
            predictions_by_image[key] = predictions
            for prediction in predictions:
                prediction_records.append({
                    "image": key,
                    "box": prediction.get("box_xyxy", prediction.get("box")),
                    "score": float(prediction.get("detector_score", 0.0)),
                })
            stream.write(json.dumps({"image": str(path), "image_key": key, "predictions": predictions}, default=float) + "\n")
            if index == 1 or index % 100 == 0 or index == len(paths):
                print(f"[{dataset_name}] {index}/{len(paths)} images", flush=True)

    elapsed = time.perf_counter() - started
    report: Dict[str, Any] = {
        "description": spec["description"],
        "image_root": str(image_root),
        "image_count": len(paths),
        "ground_truth_available": ground_truth is not None,
        "prediction_file": str(prediction_path),
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / max(len(paths), 1),
        "score_threshold": score_threshold,
        "mode": mode,
        "prediction_metrics": prediction_metrics(predictions_by_image),
    }
    if ground_truth is not None:
        report["detection_metrics"] = detection_metrics(keys, ground_truth, prediction_records)
    else:
        report["detection_metrics"] = None
        report["note"] = "No ground-truth bounding-box or identity annotation is present; accuracy metrics are unavailable."
    return report


def _percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def aggregate_detection_metrics(datasets: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Combine count-based metrics and macro-average AP over annotated sets."""
    annotated = [item["detection_metrics"] for item in datasets.values() if item.get("detection_metrics")]
    if not annotated:
        return {"annotated_datasets": 0}

    totals = {
        field: sum(int(metrics.get(field, 0)) for metrics in annotated)
        for field in (
            "images", "ground_truth_faces", "predicted_faces", "true_positives",
            "false_positives", "false_negatives",
        )
    }
    precision = totals["true_positives"] / max(
        totals["true_positives"] + totals["false_positives"], 1
    )
    recall = totals["true_positives"] / max(
        totals["true_positives"] + totals["false_negatives"], 1
    )
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    result: Dict[str, Any] = {
        "annotated_datasets": len(annotated),
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_ap50": sum(float(metrics.get("ap50", 0.0)) for metrics in annotated) / len(annotated),
    }
    for field in ("easy_ap50", "medium_ap50", "hard_ap50"):
        values = [metrics[field] for metrics in annotated if field in metrics]
        if values:
            result[f"macro_{field}"] = sum(values) / len(values)
    return result


def write_reports(report: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    csv_fields = [
        "dataset", "images", "ground_truth_faces", "predicted_faces", "true_positives",
        "false_positives", "false_negatives", "precision", "recall", "f1", "ap50",
        "easy_ap50", "medium_ap50", "hard_ap50", "known_identity_rate",
        "unknown_rejection_rate", "image_detection_rate", "mean_detections_per_image",
        "elapsed_seconds", "ground_truth_available",
    ]
    with (output_dir / "evaluation_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for name, item in report["datasets"].items():
            detection = item.get("detection_metrics") or {}
            recognition = item.get("prediction_metrics") or {}
            row = {field: "" for field in csv_fields}
            row.update({
                "dataset": name,
                "ground_truth_available": item["ground_truth_available"],
                "elapsed_seconds": f"{item['elapsed_seconds']:.3f}",
                **{field: detection.get(field, "") for field in (
                    "images", "ground_truth_faces", "predicted_faces", "true_positives",
                    "false_positives", "false_negatives", "precision", "recall", "f1",
                    "ap50", "easy_ap50", "medium_ap50", "hard_ap50",
                    "image_detection_rate", "mean_detections_per_image",
                )},
                **{field: recognition.get(field, "") for field in (
                    "known_identity_rate", "unknown_rejection_rate",
                )},
            })
            writer.writerow(row)
        aggregate = report["aggregate_detection_metrics"]
        if aggregate.get("annotated_datasets", 0):
            aggregate_row = {field: "" for field in csv_fields}
            aggregate_row.update({
                "dataset": "AGGREGATE_ANNOTATED_DATASETS",
                "ground_truth_available": True,
                **{field: aggregate.get(field, "") for field in (
                    "images", "ground_truth_faces", "predicted_faces", "true_positives",
                    "false_positives", "false_negatives", "precision", "recall", "f1",
                )},
                "ap50": aggregate.get("macro_ap50", ""),
            })
            writer.writerow(aggregate_row)

    lines = [
        "# Face Pipeline Evaluation Report",
        "",
        f"- Inference mode: `{report['mode']}`",
        f"- Detector threshold: `{report['score_threshold']}`",
        f"- Matching criterion: IoU >= `{IOU_THRESHOLD:.2f}`",
        "",
        "The supplied benchmark annotations contain face boxes, not person identity pairs. "
        "Therefore precision, recall, F1, and AP are localization metrics. Known/unknown "
        "rates are operational recognition statistics and are not identity accuracy.",
        "",
        "## Per-dataset results",
        "",
        "| Dataset | Images | GT faces | Predictions | Precision | Recall | F1 | AP@0.50 | Known rate | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in report["datasets"].items():
        detection = item.get("detection_metrics") or {}
        recognition = item.get("prediction_metrics") or {}
        lines.append(
            f"| {name} | {detection.get('images', item.get('image_count', 'n/a'))} "
            f"| {detection.get('ground_truth_faces', 'n/a')} | {recognition.get('predictions', 'n/a')} "
            f"| {_percent(detection.get('precision'))} | {_percent(detection.get('recall'))} "
            f"| {_percent(detection.get('f1'))} | {_percent(detection.get('ap50'))} "
            f"| {_percent(recognition.get('known_identity_rate'))} "
            f"| {'evaluated' if item['ground_truth_available'] else 'no ground truth'} |"
        )
    aggregate = report["aggregate_detection_metrics"]
    if aggregate.get("annotated_datasets", 0):
        lines.extend([
            "",
            "## Aggregate localization result",
            "",
            f"Across {aggregate['annotated_datasets']} annotated datasets, the micro-averaged "
            f"precision is **{_percent(aggregate['precision'])}**, recall is "
            f"**{_percent(aggregate['recall'])}**, F1 is **{_percent(aggregate['f1'])}**, "
            f"and the macro-averaged AP@0.50 is **{_percent(aggregate['macro_ap50'])}**.",
        ])
    lines.extend(["", "## Notes", ""])
    for name, item in report["datasets"].items():
        if item.get("note"):
            lines.append(f"- **{name}:** {item['note']}")
        detection = item.get("detection_metrics") or {}
        if "easy_ap50" in detection:
            lines.append(
                f"- **{name} difficulty AP:** easy {_percent(detection['easy_ap50'])}, "
                f"medium {_percent(detection['medium_ap50'])}, hard {_percent(detection['hard_ap50'])}."
            )
    (output_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one detailed evaluation over all supplied benchmark datasets")
    parser.add_argument("--data_root", default="./Evaluation", help="Evaluation dataset root")
    parser.add_argument("--weights_dir", default="./weights", help="Model checkpoint directory")
    parser.add_argument("--gallery_dir", default="./Face_Dataset/name_label", help="Identity gallery for full inference")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Report and prediction output directory")
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), help="Subset to evaluate; default is every available dataset")
    parser.add_argument("--mode", choices=("full", "detector"), default="full", help="Run the complete pipeline or only the detector")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--score_threshold", type=float, default=0.05, help="Detector score cutoff")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from inference import load_pipeline, resolve_device, resolve_detector_checkpoint
    from models.detector import FasterRCNNFaceDetector

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    selected = args.datasets or list(DATASETS)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}; score threshold: {args.score_threshold}")

    detector = None
    pipeline = None
    if args.mode == "full":
        pipeline = load_pipeline(args.weights_dir, args.gallery_dir, device)
    else:
        checkpoint = resolve_detector_checkpoint(Path(args.weights_dir), None)
        detector = FasterRCNNFaceDetector(
            weights_dir=args.weights_dir,
            detector_checkpoint=str(checkpoint) if checkpoint else None,
        ).to(device).eval()

    datasets: Dict[str, Any] = OrderedDict()
    for dataset_name in selected:
        spec = DATASETS[dataset_name]
        image_root = data_root / str(spec["image_path"])
        if not image_root.is_dir():
            print(f"Skipping {dataset_name}: missing {image_root}")
            continue
        print(f"\n=== {dataset_name}: {spec['description']} ===")
        datasets[dataset_name] = run_dataset(
            dataset_name,
            spec,
            data_root,
            output_dir,
            device,
            args.mode,
            args.score_threshold,
            detector,
            pipeline,
        )

    if not datasets:
        raise SystemExit("No requested evaluation datasets were found")
    report = {
        "mode": args.mode,
        "device": str(device),
        "score_threshold": args.score_threshold,
        "iou_threshold": IOU_THRESHOLD,
        "datasets": datasets,
        "aggregate_detection_metrics": aggregate_detection_metrics(datasets),
        "identity_accuracy_note": (
            "Identity accuracy requires person-identity labels or genuine/impostor pairs. "
            "The supplied datasets provide face localization annotations only."
        ),
    }
    write_reports(report, output_dir)
    print(f"\nSaved detailed reports under {output_dir}")
    print(f"  Markdown: {output_dir / 'evaluation_report.md'}")
    print(f"  CSV:      {output_dir / 'evaluation_report.csv'}")
    print(f"  JSON:     {output_dir / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
