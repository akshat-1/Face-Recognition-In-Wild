# Face Recognition in the Wild

This repository implements a multi-phase face pipeline for unconstrained
images. It detects faces in a complete scene, converts each detected region to
a canonical face crop, extracts a 512-dimensional identity embedding, and
optionally reports an identity, attributes, occlusion, and confidence.

The system is **open-set**: a face is returned as `unknown` when it does not
match an enrolled gallery with sufficient reconstruction quality and margin.
The trained neural-network checkpoints do not contain the identity names used
for deployment. Names are attached at inference time by enrolling a labelled
gallery with the same recognition backbone.

## Quick start

Run one image:

```bash
python inference.py \
  --image Test_dataser/image11s.jpg \
  --weights_dir weights \
  --gallery_dir Face_Dataset/name_label \
  --output_dir inference_results \
  --device auto \
  --score_threshold 0.05
```

Run every supported image below a directory:

```bash
python inference.py \
  --input_dir Test_dataser \
  --weights_dir weights \
  --gallery_dir Face_Dataset/name_label \
  --output_dir inference_results \
  --device cuda \
  --score_threshold 0.05
```

The two input options are mutually exclusive. The command writes:

```text
inference_results/
├── predictions.jsonl       # one JSON object per input image
└── annotated/              # copies with predicted boxes and labels
```

## Benchmark evaluation

Run the complete pipeline once over every dataset under `Evaluation/`:

```bash
python evaluate_all.py \
  --data_root Evaluation \
  --weights_dir weights \
  --gallery_dir Face_Dataset/name_label \
  --output_dir evaluation_results \
  --device cuda \
  --mode full \
  --score_threshold 0.05
```

The runner automatically evaluates WIDER FACE, UFDD, AFW, and DARKFACE when
their folders are present. It writes `evaluation_report.md` for paper tables,
`evaluation_report.csv`, `evaluation_report.json`, and one JSONL prediction
file per dataset under `evaluation_results/`. Annotated datasets report
precision, recall, F1, AP@IoU=0.50, image coverage, and WIDER-style
easy/medium/hard AP. Full mode also reports known-versus-unknown recognition
rates and attribute/occlusion operating counts.

The supplied benchmark annotations contain face boxes but no person identity
pairs. Consequently, the detection metrics are the accuracy measures that can
be reported for these datasets; known/unknown rates are diagnostic and are not
identity-recognition accuracy. DARKFACE is reported as inference-only unless
ground-truth annotations are added.

Use `--mode detector` for a faster localization-only benchmark. Genuine versus
impostor identity verification requires a pair protocol and can be evaluated
with `evaluate_verification.py` after exporting embeddings.

For a detector-only diagnostic, which is useful when recognition returns no
identity, run:

```bash
python detector_inference.py \
  --image Test_dataser/image11s.jpg \
  --weights_dir weights \
  --score_threshold 0.05 \
  --overlay_dir detector_results \
  --device cuda
```

## Architecture

The training phases produce compatible artifacts rather than one monolithic
network:

```text
scene image
    │
    ▼
Phase 0: Faster R-CNN ResNet-50 FPN face detector
    │  face boxes + detector scores
    ▼
crop, clip to image, resize/alignment to 112 × 112
    │                         └──────────────┐
    ▼                                        ▼
Phase 1 or Phase 3                   Phase 2 ANet parser
IResNet-100                         40 attributes + occlusion map
512-D L2-normalized embedding
    │
    ▼
gallery enrollment + DDRC/LISTA open-set classifier
    │
    ▼
identity or unknown, recognition score, residuals, confidence
```

### Phase 0: face localization

`models/detector.py` uses the trained `TorchvisionFaceDetector` configured as:

- Faster R-CNN with a ResNet-50 FPN;
- two classes: background and face;
- FPN anchor sizes `16, 32, 64, 128, 256` with aspect ratios
  `0.5, 1.0, 2.0`;
- detector resize limits of `min_size=800` and `max_size=1333` during the
  completed AQUA training run;
- class-agnostic treatment of every foreground YOLO annotation as a face.

At inference, the scene is kept at its native resolution. Faster R-CNN applies
its own resize policy. Resizing a complete scene to `224 × 224` before calling
the detector makes small faces disappear and should be avoided.

The detector returns boxes and scores. The Faster R-CNN checkpoint does not
produce facial landmarks, so its landmark tensor is zero-filled and the later
pipeline uses box-resize alignment.

### Phase 1: identity representation

The recognition model is an InsightFace-style IResNet-100:

- four residual stages with channel widths `64, 128, 256, 512`;
- convolutional stem, batch normalization, PReLU, residual blocks, and a
  `512 × 7 × 7` projection head;
- final `512`-dimensional batch-normalized feature;
- L2 normalization of the output embedding.

During training, the embedding is optimized with `BroadFaceCurricularLoss`:

- angular/cosine classification with scale `64` and margin `0.50`;
- CurricularFace-style difficulty tracking;
- a detached BroadFace memory queue of previous embeddings.

The loss and queue are training mechanisms. Final inference uses the backbone
checkpoint and the separately enrolled gallery.

### Phase 2: attributes and occlusion

`ANetAttributeParser` processes the 112-pixel face crop using two paths:

1. a global convolutional path;
2. an upper-face/local path, or a landmark-guided local path when landmarks
   are available.

The paths are fused into a 256-channel feature map and produce:

- 40 CelebA-style binary attributes;
- a `7 × 7` spatial occlusion map;
- an occlusion flag.

The attributes are reported in parallel with identity. They do not directly
change the identity label.

### Phase 3: semi-supervised adaptation

Phase 3 adapts the recognition backbone using detector-approved unlabeled face
crops:

1. Phase 0 detects faces in scenes, if `unlabeled_mode=detected_scenes`;
2. the existing recognition backbone creates embeddings;
3. a KNN/GCN-style graph links similar embeddings;
4. high-confidence connected components become pseudo-labels;
5. the backbone is fine-tuned jointly with labelled data and weighted
   pseudo-labelled data.

The graph is primarily a training-time clustering/adaptation mechanism. During
normal final prediction, `phase3_adapted_backbone.pt` replaces the Phase 1
backbone. The query is then classified against the enrolled gallery.

The current cached Phase 3 artifact may contain pseudo-label metadata without
a serialized graph predictor. This is valid for inference because the adapted
backbone is the artifact used for the query embedding.

### Gallery classifier and open-set decision

`DDRCClassifier` builds a dictionary from enrolled gallery embeddings. Its
LISTA-style sparse solver reconstructs a query embedding from class atoms and
computes a residual for every class. A class is accepted only when:

- its residual is below the residual threshold; and
- the gap between the best and second-best class is above the margin
  threshold.

Otherwise the result is `unknown`. The reported recognition score is derived
from residual quality and the best-vs-second-best margin.

The gallery is not part of the neural checkpoint. It is recomputed or loaded
from `weights/gallery_embeddings.pt` using a signature of the gallery files.

## Where the phases meet during inference

For each input image:

1. `inference.py` loads Phase 0, Phase 3/Phase 1, Phase 2, and the gallery.
2. The native-resolution image is converted to the detector tensor range
   `[-1, 1]`.
3. Phase 0 returns candidate face boxes.
4. Each box is clipped and cropped from the original tensor.
5. The crop is resized to `112 × 112`; valid landmarks would enable similarity
   alignment, but the trained Faster R-CNN detector currently supplies none.
6. The selected recognition backbone creates one normalized 512-D embedding.
7. The DDRC gallery classifier returns an identity or `unknown`.
8. Phase 2 analyzes the same crop for attributes and occlusion.
9. The final confidence is `detector_score × recognition_score`.

The checkpoints are not averaged. They are connected by the detected crop and
the shared face embedding.

## Checkpoints

The inference loader searches `--weights_dir` in this order:

| Artifact | Role |
|---|---|
| `face_detector.pt` or `phase0_detector/face_detector.pt` | Phase 0 detector |
| `phase3_adapted_backbone.pt` | preferred Phase 3 recognition backbone |
| `recognition_backbone.pt` | Phase 1 recognition backbone |
| `latest_backbone.pt` | resumable/older Phase 1 artifact |
| `phase1_backbone_curricular.pt` | Phase 1 compatibility artifact |
| `phase2_anet_attributes.pt` | Phase 2 ANet attributes and occlusion |
| `phase3_semi_gcn_predictor.pt` | Phase 3 metadata and optional graph state |
| `gallery_embeddings.pt` | generated gallery cache |

If both Phase 1 and Phase 3 backbone files exist, Phase 3 wins because it is
searched first. To compare Phase 1 and Phase 3 predictions, use separate
checkpoint directories containing the desired backbone artifact.

Weights are intentionally ignored by Git. Keep them in the local `weights/`
directory or download them from the training system.

## Inference configuration

`inference.py --help` is the authoritative CLI reference. Current options:

| Option | Default | Purpose |
|---|---:|---|
| `--image` | none | one image; mutually exclusive with `--input_dir` |
| `--input_dir` | none | recursively process image files |
| `--weights_dir` | `./weights` | checkpoint bundle |
| `--detector_checkpoint` | auto | explicit Phase 0 checkpoint |
| `--gallery_dir` | `./Face_Dataset/name_label` | labelled enrollment gallery |
| `--output_dir` | `./inference_results` | output directory |
| `--output_json` | `predictions.jsonl` | JSONL filename |
| `--score_threshold` | `0.05` | detector score cutoff |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |

### Threshold tuning

`--score_threshold` controls **face detection**, not identity recognition.

- Start at `0.05` for blurred, dark, small, or occluded faces.
- Try `0.10–0.20` to reduce weak false-positive boxes.
- Use `0.30–0.40` when precision is more important than recall.
- If a detector-only run produces no box, lowering the recognition/gallery
  settings cannot help; lower this detector threshold or improve the input.

The Phase 3 training default `unlabeled_score_threshold=0.75` is deliberately
different. It controls which detections are trusted as pseudo-training crops;
it is not the recommended final-prediction threshold for difficult images.

### Improving identity predictions

If boxes are present but all identities are `unknown`:

1. Confirm that `--gallery_dir` exists and contains one identity folder per
   person.
2. Enroll several clear, varied crops per person, including pose and lighting
   conditions expected at inference.
3. Ensure gallery images are face crops, not complete scenes.
4. Keep gallery and query preprocessing consistent; both use 112-pixel crops
   and the same normalized transform.
5. Delete `weights/gallery_embeddings.pt` if a gallery was changed in a way
   not reflected by file size or modification time, then rerun enrollment.
6. Compare the Phase 3-adapted and Phase 1 backbone using separate weight
   directories.

An unknown result can be correct: the classifier is intentionally open-set and
does not force every detected face into the nearest enrolled identity.

The current open-set gates are code-level settings in
`models/ddrc_solver.py`: `residual_threshold=0.55` and
`margin_threshold=0.10`. The pipeline currently uses these defaults and does
not expose them as CLI flags. A larger residual threshold or smaller margin
threshold accepts more weak matches; that can improve recall while increasing
false identities. Tune them only against a held-out known/unknown set.

The gallery dictionary used by `inference.py` currently stores one atom per
identity (`atoms_per_class=1`). Adding more varied gallery samples improves
coverage, but changing the atom-count policy requires a small code-level
configuration change and rebuilding the gallery cache.

### Improving difficult detections

- Use the original-resolution input file; do not pre-resize a scene to a small
  thumbnail.
- Prefer a sharp, adequately exposed image.
- For a very large image, the detector's trained maximum side length already
  limits internal processing to 1333 pixels.
- Inspect `detector_inference.py` overlays before diagnosing recognition.
- Use a higher threshold to suppress false positives and a lower threshold to
  recover weak faces; these are competing objectives.

Changing the detector architecture, anchor sizes, detector resize limits, or
recognition input size after training is not a weight-free quality adjustment.
Those changes alter the model contract and normally require retraining or a
compatible checkpoint.

## Training configuration

The dataclass defaults live in `config.py`; command-line arguments override
the relevant values in `train.py`.

### Dataset configuration

| Setting | Meaning |
|---|---|
| `train_data_dir` / `--data_dir` | identity-labelled face crops |
| `bb_data_dir` / `--bb_dir` | YOLO/Darknet face-box dataset |
| `unlabeled_data_dir` / `--unlabeled_dir` | unlabeled crops or scenes |
| `celeba_dir` / `--celeba_dir` | CelebA images and attribute annotations |
| `wider_identity_data_dir` / `--wider_dir` | optional folder-labelled WIDER crops |
| `image_size` | fixed at `(112, 112)` for recognition and attributes |
| `unlabeled_mode` / `--unlabeled_mode` | `detected_scenes` or `face_crops` |
| `unlabeled_score_threshold` | detector cutoff for Phase 3 crop creation |
| `num_workers` | DataLoader worker count |

The completed AQUA run used named identity crops plus folder-labelled WIDER
crops for recognition, and detector-approved scenes for Phase 3.

### Model and loss configuration

| Setting | Default | Effect |
|---|---:|---|
| `backbone_type` / `--backbone` | `iresnet100` | identity backbone architecture |
| `embedding_dim` | `512` | recognition vector size |
| `fp16` / `--fp16` | CUDA-enabled | mixed-precision training |
| `detector_score_threshold` | `0.40` | configuration value for detector workflows |
| `recognition_alignment` | `resize` | `resize` or `landmarks` |
| `pose_gate_degrees` | `35` | frontalizer pose gate when a compatible module exists |
| `scale` | `64` | angular-margin loss scale |
| `margin` | `0.50` | angular-margin loss margin |
| `queue_size` | `32768` | BroadFace memory size |
| `lambda_semi` | `0.5` | Phase 3 pseudo-label loss weight |
| `pseudo_label_threshold` | `0.75` | graph/cosine pseudo-label confidence |
| `atoms_per_class` | `20` in training config | gallery dictionary atoms per identity |

The recognizer and attribute parser require 112×112 inputs. Changing this
size, embedding dimension, or backbone type requires a newly compatible
checkpoint; it is not a safe inference-only tuning knob.

### Training command examples

Run the complete configured sequence locally:

```bash
python train.py \
  --phase all \
  --bb_dir Face_Dataset/bb_label \
  --data_dir Face_Dataset/name_label \
  --unlabeled_dir Face_Dataset/unlabeled \
  --celeba_dir path/to/img_align_celeba \
  --checkpoint_dir weights \
  --backbone iresnet100 \
  --unlabeled_mode detected_scenes \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.1 \
  --fp16
```

Run only the detector:

```bash
python train.py --phase detector \
  --bb_dir Face_Dataset/bb_label \
  --checkpoint_dir weights \
  --detector_epochs 10 \
  --detector_min_size 800 \
  --detector_max_size 1333
```

Resume Phase 1 from `latest_backbone.pt`:

```bash
python train.py --phase recognition --resume \
  --data_dir Face_Dataset/name_label \
  --unlabeled_dir Face_Dataset/unlabeled \
  --celeba_dir path/to/img_align_celeba \
  --checkpoint_dir weights
```

Run Phase 3 using an already completed recognition checkpoint:

```bash
python train.py --phase adapt \
  --data_dir Face_Dataset/name_label \
  --unlabeled_dir Face_Dataset/unlabeled \
  --checkpoint_dir weights \
  --unlabeled_mode detected_scenes
```

`--phase recognition` runs Phase 1 and, when `--celeba_dir` is supplied,
Phase 2. It does not start Phase 3. Use `--phase all` for the complete
sequence, or `--phase adapt` for Phase 3 after recognition already exists.

The `train_aqua_full.cmd` PBS script supplies the cluster paths, CUDA setup,
distributed launch, and the completed-run settings. Set path variables such as
`AQUA_PROJECT_ROOT`, `AQUA_DATA_ROOT`, and `AQUA_CHECKPOINT_DIR` when submitting
on AQUA. Do not point a new job at an active job's checkpoint directory.

## Dataset layout

The expected high-level layout is:

```text
Face_Dataset/
├── bb_label/                 # YOLO images and labels for Phase 0
├── name_label/               # identity folders for Phase 1/gallery
└── unlabeled/                # scenes/crops and optional WIDER data
    └── masked-face-detection-wider-dataset/
```

For identity training and enrollment, each identity should have a directory
containing face crops. For detector training, each image needs a corresponding
YOLO text file. Every foreground YOLO row is treated as a face regardless of
its source class ID.

## Output schema

Each JSONL record has an image path and a `predictions` list. A prediction can
include:

```text
box / box_xyxy          original-image pixel coordinates
identity                enrolled name or "unknown"
is_unknown              open-set decision
detector_score          Phase 0 score
recognition_score       gallery score
confidence              detector_score × recognition_score
attributes              40 attribute booleans when Phase 2 is loaded
is_occluded             ANet occlusion flag
pose                    geometry signal; invalid when landmarks are absent
aligned                 whether landmark alignment was used
frontalized             whether a compatible frontalizer was used
residuals               per-class DDRC residuals
```

## Verification and troubleshooting

Check the code and command interfaces:

```bash
python -m py_compile inference.py detector_inference.py train.py
python inference.py --help
python detector_inference.py --help
python -m unittest tests.test_robustness tests.test_regressions -v
```

Common symptoms:

| Symptom | Likely cause | Check |
|---|---|---|
| No predictions | detector threshold or checkpoint issue | run `detector_inference.py` |
| Boxes but all `unknown` | missing/weak gallery | verify `--gallery_dir` and enrollment |
| Recognition loader fails on Phase 3 file | incompatible old artifact | use a schema-2 checkpoint and current loader |
| Attributes absent | Phase 2 checkpoint not found | check `phase2_anet_attributes.pt` |
| CUDA requested but unavailable | wrong runtime/node | use `--device auto` or `cpu` |
| Slow first run | gallery embeddings are being computed | reuse `weights/gallery_embeddings.pt` |

## Important limitations

- The detector is class-agnostic: it detects face regions, not identities.
- Faster R-CNN supplies no landmarks in this implementation, so landmark pose
  and landmark alignment are unavailable for normal final inference.
- The gallery is deployment state and must be built with the same recognition
  preprocessing as the query images.
- Phase 3 graph clustering is not a pixel segmentation model and does not
  directly draw boxes.
- Lowering the detector threshold increases recall but can also introduce
  false-positive regions; inspect detector overlays before changing gallery
  thresholds.
