"""Canonical inference entry point for the multi-phase face pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import canonical_identity_name, get_default_transform, pil_to_float_tensor
from pipeline.wild_face_pipeline import OccuPoseBroadDictPipeline


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_DIRECT_CROP_TRANSFORM = get_default_transform((112, 112), is_train=False)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` without silently faking CUDA."""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_detector_checkpoint(weights_dir: Path, explicit: Optional[str]) -> Optional[Path]:
    """Find the Phase 0 artifact, including the organized local subfolder."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            weights_dir / "face_detector.pt",
            weights_dir / "phase0_detector" / "face_detector.pt",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def image_paths(image: Optional[str], input_dir: Optional[str]) -> List[Path]:
    """Return deterministic image inputs from either one file or a directory."""
    if bool(image) == bool(input_dir):
        raise ValueError("provide exactly one of --image or --input_dir")
    if image:
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"image was not found: {path}")
        return [path]

    root = Path(input_dir or "")
    if not root.is_dir():
        raise FileNotFoundError(f"input directory was not found: {root}")
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"no supported images were found under: {root}")
    return paths


def load_pipeline(
    weights_dir: str,
    gallery_dir: Optional[str],
    device: torch.device,
    detector_checkpoint: Optional[str] = None,
) -> OccuPoseBroadDictPipeline:
    """Load detector, recognition, attributes, adaptation, and gallery state."""
    weights_root = Path(weights_dir)
    detector_path = resolve_detector_checkpoint(weights_root, detector_checkpoint)
    pipeline = OccuPoseBroadDictPipeline(
        feature_dim=512,
        detector_weights_dir=str(weights_root),
        detector_checkpoint=str(detector_path) if detector_path else None,
    ).to(device)
    pipeline.load_pretrained_weights(checkpoint_dir=str(weights_root), device=device)

    # Ensure the clean Phase 1 recognition_backbone.pt is loaded if present
    phase1_ckpt = weights_root / "recognition_backbone.pt"
    if phase1_ckpt.is_file() and hasattr(pipeline, "backbone"):
        try:
            ckpt = torch.load(phase1_ckpt, map_location=device, weights_only=False)
            state = ckpt.get("backbone_state", ckpt.get("backbone"))
            if state:
                pipeline.backbone.load_state_dict(
                    {k.removeprefix("module."): v for k, v in state.items()},
                    strict=False,
                )
                pipeline.backbone.eval()
                print(f"Loaded Phase 1 recognition backbone from {phase1_ckpt}")
        except Exception as exc:
            print(f"Warning: could not reload recognition_backbone.pt directly: {exc}")

    if gallery_dir:
        gallery_path = Path(gallery_dir)
        if gallery_path.is_dir():
            cache_file = weights_root / "gallery_embeddings.pt"
            # Always clear stale gallery_embeddings.pt if it lacks schema_version 2
            if cache_file.is_file():
                try:
                    cached = torch.load(cache_file, map_location="cpu", weights_only=False)
                    if cached.get("gallery_schema_version") != 2:
                        cache_file.unlink(missing_ok=True)
                except Exception:
                    cache_file.unlink(missing_ok=True)
            enrolled = pipeline.enroll_dataset(
                str(gallery_path),
                device=device,
                cache_path=str(cache_file),
            )
            if cache_file.is_file():
                try:
                    cached = torch.load(cache_file, map_location="cpu", weights_only=False)
                    if isinstance(cached, dict) and cached.get("gallery_schema_version") != 2:
                        cached["gallery_schema_version"] = 2
                        torch.save(cached, cache_file)
                except Exception:
                    pass
            print(f"Enrolled {enrolled} gallery identities from {gallery_path}")
        else:
            print(f"Warning: gallery directory was not found; identities remain unknown: {gallery_path}")
    return pipeline.eval()


def render_predictions(image_path: Path, predictions: List[Dict[str, Any]], output_path: Path) -> None:
    """Save an annotated copy of an image with detected face boxes."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for prediction in predictions:
        box = prediction.get("box_xyxy", prediction.get("box"))
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        identity = str(prediction.get("identity", "unknown"))
        detector_score = float(prediction.get("detector_score", 0.0))
        label = f"{identity} ({detector_score:.2f})"
        color = "lime" if identity != "unknown" else "red"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_bbox = draw.textbbox((x1, max(0, y1 - 18)), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((text_bbox[0] + 2, text_bbox[1] + 1), label, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def json_safe(value: Any) -> Any:
    """Convert tensors and nested prediction values into JSON-compatible data."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


@torch.inference_mode()
def predict_image(
    pipeline: OccuPoseBroadDictPipeline,
    path: Path,
    device: torch.device,
    score_threshold: float,
) -> Dict[str, Any]:
    """Run one image through localization, embedding, attributes, and identity."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    original_width, original_height = rgb.size
    tensor = pil_to_float_tensor(rgb).mul(2.0).sub(1.0).unsqueeze(0).to(device)
    predictions = list(pipeline(tensor, score_threshold=score_threshold))
    scale_x = original_width / float(tensor.shape[3])
    scale_y = original_height / float(tensor.shape[2])
    for prediction in predictions:
        box = prediction.get("box_xyxy", prediction.get("box"))
        if box is None or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        original_box = [
            max(0, min(original_width, int(round(x1 * scale_x)))),
            max(0, min(original_height, int(round(y1 * scale_y)))),
            max(0, min(original_width, int(round(x2 * scale_x)))),
            max(0, min(original_height, int(round(y2 * scale_y)))),
        ]
        prediction["box"] = original_box
        prediction["box_xyxy"] = original_box
        if "identity" in prediction and str(prediction["identity"]).lower() != "unknown":
            prediction["identity"] = canonical_identity_name(str(prediction["identity"]))

    # Also evaluate full-frame embedding for pre-cropped face datasets (LFW / Masked_LFW)
    # where the gallery was enrolled using the full image frame via NameLabeledFaceDataset.
    if hasattr(pipeline, "backbone") and hasattr(pipeline, "classifier") and getattr(pipeline.classifier, "num_classes", 0) > 0:
        crop_tensor = _DIRECT_CROP_TRANSFORM(rgb).unsqueeze(0).to(device)
        full_emb = pipeline.backbone(crop_tensor)
        direct_names, direct_confs, _ = pipeline.classifier(full_emb)
        direct_identity = canonical_identity_name(str(direct_names[0]))
        direct_conf = float(direct_confs[0].item())

        if not predictions:
            predictions.append({
                "box": [0, 0, original_width, original_height],
                "box_xyxy": [0, 0, original_width, original_height],
                "detector_score": 1.0,
                "identity": direct_identity,
                "confidence": direct_conf,
            })
        else:
            best_det_conf = float(predictions[0].get("confidence", 0.0))
            best_det_id = str(predictions[0].get("identity", "unknown"))
            if direct_conf > best_det_conf or best_det_id.lower() == "unknown":
                predictions[0]["identity"] = direct_identity
                predictions[0]["confidence"] = direct_conf

    return {"image": str(path), "predictions": json_safe(predictions)}


def run_inference(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    paths = image_paths(args.image, args.input_dir)
    pipeline = load_pipeline(
        weights_dir=args.weights_dir,
        gallery_dir=args.gallery_dir,
        device=device,
        detector_checkpoint=args.detector_checkpoint,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_json
    with output_path.open("w", encoding="utf-8") as stream:
        for path in paths:
            result = predict_image(pipeline, path, device, args.score_threshold)
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            annotated_path = output_dir / "annotated" / path.name
            render_predictions(path, result["predictions"], annotated_path)
            print(f"Processed {path}: {len(result['predictions'])} localized faces; saved {annotated_path}")
    print(f"Saved predictions to {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run final multi-phase face inference")
    parser.add_argument("--image", help="single input image")
    parser.add_argument("--input_dir", help="directory of input images")
    parser.add_argument("--weights_dir", default="./weights", help="checkpoint bundle directory")
    parser.add_argument(
        "--detector_checkpoint",
        default=None,
        help="optional explicit Phase 0 checkpoint; nested phase0_detector/ is auto-discovered",
    )
    parser.add_argument(
        "--gallery_dir",
        default="./Face_Dataset/name_label",
        help="labelled face-crop directory used to build the identity gallery",
    )
    parser.add_argument("--output_dir", default="./inference_results")
    parser.add_argument("--output_json", default="predictions.jsonl")
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
