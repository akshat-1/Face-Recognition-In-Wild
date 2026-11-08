"""Evaluate genuine/impostor face-verification pairs.

This utility is intentionally separate from ``evaluate_all.py``.  Detection
benchmarks such as WIDER FACE, UFDD, AFW, and DARKFACE do not contain person
identity pairs, so they cannot produce identity-verification accuracy.  Use
this script when a benchmark supplies an embeddings JSONL file and a pairs
file (for example, an LFW-style protocol).

The embeddings file accepts either of these records::

    {"image": "person/person_0001.jpg", "embedding": [0.1, ...]}
    {"image": "person/person_0001.jpg", "embedding": [0.1, ...], "label": "person"}

The pairs file supports standard LFW lines (three fields for genuine pairs,
four for impostor pairs) and simple ``image_a image_b label`` lines where the
label is ``1``/``same`` or ``0``/``different``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def calculate_roc(
    thresholds: Sequence[float],
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    actual_issame: Sequence[bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return TPR, FPR, and accuracy at every squared-distance threshold."""
    if embeddings1.shape != embeddings2.shape:
        raise ValueError("paired embedding arrays must have the same shape")
    labels = np.asarray(actual_issame, dtype=bool)
    if labels.shape[0] != embeddings1.shape[0]:
        raise ValueError("the number of pair labels must equal the number of pairs")
    distances = np.sum(np.square(embeddings1 - embeddings2), axis=1)
    tpr, fpr, accuracy = [], [], []
    for threshold in thresholds:
        accepted = distances < float(threshold)
        tp = np.sum(accepted & labels)
        fp = np.sum(accepted & ~labels)
        tn = np.sum(~accepted & ~labels)
        fn = np.sum(~accepted & labels)
        tpr.append(tp / max(tp + fn, 1))
        fpr.append(fp / max(fp + tn, 1))
        accuracy.append((tp + tn) / max(len(labels), 1))
    return np.asarray(tpr), np.asarray(fpr), np.asarray(accuracy)


def load_embeddings(path: Path) -> Dict[str, np.ndarray]:
    """Load and L2-normalize embeddings keyed by normalized image path."""
    result: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "image" not in record or "embedding" not in record:
                raise ValueError(f"embedding record on line {line_number} needs image and embedding")
            key = Path(str(record["image"])).as_posix()
            vector = np.asarray(record["embedding"], dtype=np.float32).reshape(-1)
            if vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError(f"invalid embedding for {key!r} on line {line_number}")
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError(f"zero embedding for {key!r} on line {line_number}")
            if key in result:
                raise ValueError(f"duplicate embedding key: {key!r}")
            result[key] = vector / norm
    if not result:
        raise ValueError(f"no embeddings found in {path}")
    dimensions = {vector.size for vector in result.values()}
    if len(dimensions) != 1:
        raise ValueError(f"embeddings have inconsistent dimensions: {sorted(dimensions)}")
    return result


def _resolve_key(raw: str, embeddings: Mapping[str, np.ndarray]) -> str:
    normalized = Path(raw).as_posix()
    if normalized in embeddings:
        return normalized
    matches = [key for key in embeddings if Path(key).name == Path(normalized).name]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"could not resolve pair image {raw!r} in the embeddings file")


def _label(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "same", "genuine", "yes"}:
        return True
    if value in {"0", "false", "different", "impostor", "no"}:
        return False
    raise ValueError(f"unsupported pair label: {value!r}")


def load_pairs(path: Path, embeddings: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Load explicit pairs or standard LFW pair lines."""
    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_lines:
        raise ValueError(f"pairs file is empty: {path}")
    start = 1 if len(raw_lines[0].split()) == 1 and raw_lines[0].split()[0].isdigit() else 0
    pairs: List[Dict[str, Any]] = []
    for line_number, line in enumerate(raw_lines[start:], start=start + 1):
        fields = line.split()
        if len(fields) == 3 and fields[1].isdigit() and fields[2].isdigit():
            # Standard LFW genuine-pair line: name index1 index2.
            name, first_index, second_index = fields
            first = f"{name}/{name}_{int(first_index):04d}.jpg"
            second = f"{name}/{name}_{int(second_index):04d}.jpg"
            same = True
        elif len(fields) == 3:
            first, second, label = fields
            same = _label(label)
        elif len(fields) == 4:
            first, first_index, second, second_index = fields
            first = f"{first}/{first}_{int(first_index):04d}.jpg"
            second = f"{second}/{second}_{int(second_index):04d}.jpg"
            same = False
        elif len(fields) == 2:
            first, second = fields
            same = True
        else:
            raise ValueError(f"unsupported pair format on line {line_number}: {line!r}")
        try:
            first_key = _resolve_key(first, embeddings)
            second_key = _resolve_key(second, embeddings)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"could not resolve pair on line {line_number}: {line!r}") from exc
        pairs.append({"first": first_key, "second": second_key, "same": same})
    if not pairs:
        raise ValueError(f"no pairs found in {path}")
    return pairs


def evaluate_pairs(embeddings: Mapping[str, np.ndarray], pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute threshold-selected verification metrics."""
    first = np.stack([embeddings[pair["first"]] for pair in pairs])
    second = np.stack([embeddings[pair["second"]] for pair in pairs])
    labels = np.asarray([bool(pair["same"]) for pair in pairs], dtype=bool)
    distances = np.sum(np.square(first - second), axis=1)
    thresholds = np.linspace(0.0, 4.0, 4001)
    _, fpr, accuracy = calculate_roc(thresholds, first, second, labels)
    best_index = int(np.argmax(accuracy))
    threshold = float(thresholds[best_index])
    accepted = distances < threshold
    tp = int(np.sum(accepted & labels))
    fp = int(np.sum(accepted & ~labels))
    tn = int(np.sum(~accepted & ~labels))
    fn = int(np.sum(~accepted & labels))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "pairs": len(pairs),
        "genuine_pairs": int(np.sum(labels)),
        "impostor_pairs": int(np.sum(~labels)),
        "threshold_squared_l2": threshold,
        "accuracy": float(accuracy[best_index]),
        "true_accepts": tp,
        "false_accepts": fp,
        "true_rejects": tn,
        "false_rejects": fn,
        "precision": float(precision),
        "true_accept_rate": float(recall),
        "false_accept_rate": float(1.0 - specificity),
        "mean_genuine_distance": float(np.mean(distances[labels])) if np.any(labels) else None,
        "mean_impostor_distance": float(np.mean(distances[~labels])) if np.any(~labels) else None,
        "min_observed_far": float(np.min(fpr)) if len(fpr) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate genuine/impostor face-verification pairs")
    parser.add_argument("--embeddings", required=True, help="JSONL with image and embedding fields")
    parser.add_argument("--pairs", required=True, help="Pair protocol file")
    parser.add_argument("--output_path", default="verification_results.json", help="JSON output path")
    args = parser.parse_args()
    embeddings = load_embeddings(Path(args.embeddings))
    pairs = load_pairs(Path(args.pairs), embeddings)
    metrics = evaluate_pairs(embeddings, pairs)
    Path(args.output_path).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved verification metrics to {args.output_path}")


if __name__ == "__main__":
    main()
