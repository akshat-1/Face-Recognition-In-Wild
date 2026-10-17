# OccuPose-BroadDictNet — Implemented Architecture

This document describes the code that is actually implemented in this
repository. It intentionally distinguishes implemented behavior from optional
research ideas and from claims that require external checkpoints or benchmark
evidence.

## 1. Scope and design decisions

The system recognizes enrolled identities in unconstrained images while
remaining safe when a detector, preprocessing checkpoint, gallery, or
evaluation annotation is unavailable.

The implemented production path is:

```text
RGB image
  -> face detector (YuNet and/or MTCNN)
  -> index-preserving Soft-NMS
  -> five-point similarity alignment
  -> optional checkpointed foreground/attribute preprocessing
  -> optional checkpointed pose frontalization
  -> normalized IResNet-100 or ViT-style embedding
  -> enrollment gallery and dictionary-conditioned open-set classifier
  -> known identity or unknown
```

Important safety rules:

* No detector backend silently creates a full-image face.
* No missing dataset silently becomes random training data.
* No untrained SAM, PIM, or attribute module is inserted into recognition.
* Gallery enrollment and query inference use the same crop/alignment path.
* Detection confidence and recognition confidence are reported separately.
* WIDER FACE AP is emitted only when valid ground truth covers the evaluation
  set.

The system is a research implementation, not a claim of state-of-the-art
accuracy. Accuracy must be established with the evaluation protocols in this
document and with real trained checkpoints.

## 2. Repository modules

```text
config.py                         Dataclass configuration and safety flags
dataset.py                        Fail-fast task-specific image datasets
train.py                          Phase 1 identity, Phase 2 attributes,
                                  optional Phase 3 graph adaptation
pipeline/alignment.py             Canonical five-point alignment and pose signal
pipeline/wild_face_pipeline.py   Inference, enrollment, and output contract

models/detector.py                YuNet/MTCNN facade and indexed Soft-NMS
models/backbone.py                IResNet and ViT-style embedding backbones
models/anet_attribute.py          Optional 40-label CelebA attribute parser
models/sam_segmentor.py           Optional custom foreground-mask module
models/pim_frontalizer.py         Optional frontalizer/SR module
models/ddrc_solver.py             Dictionary-conditioned sparse open-set model
models/gcn_cluster.py             GCN edge model and global pseudo-clustering

losses/curricular_loss.py         Standalone CurricularFace loss
losses/broadface_queue.py         Curricular classifier plus explicit memory queue
losses/wing_loss.py               Generic Wing Loss implementation

eval_widerface.py                 Ground-truth-required WIDER evaluator
```

## 3. Detection and alignment

### Detector contract

`models.detector.SOTAFaceDetector` supports:

* OpenCV YuNet when `weights/face_detection_yunet_2023mar.onnx` is available;
* MTCNN when `facenet-pytorch` is installed;
* both backends together in `auto` mode;
* optional image-pyramid TTA at scales `0.75`, `1.0`, and `1.25`.

The detector returns:

```text
boxes:     (N, 4) xyxy pixel coordinates
scores:    (N,) detector probabilities
landmarks: (N, 5, 2) five-point pixel coordinates
```

Every Soft-NMS operation carries source indices so a surviving box retains
the correct score and landmarks. If no usable detector is available, all three
outputs are empty. The legacy random LNet fallback is no longer used.

`FasterRCNNFaceDetector` remains as a compatibility facade, but generic COCO
Faster R-CNN is not treated as a face detector. A face-trained detector would
need its own checkpoint and benchmark report before being enabled.

### Alignment contract

`pipeline/alignment.py` uses the standard five-point 112x112 template:

```text
left eye, right eye, nose, left mouth corner, right mouth corner
```

If valid landmarks are available, the crop is transformed with a similarity
mapping. If landmarks are unavailable or degenerate, the crop is resized and
the output records `aligned=False`. Training face crops, gallery enrollment,
and query images must use this same helper.

Pose is a geometry-based signal derived from the landmarks. It is intentionally
reported as a quality signal rather than described as a calibrated 3-D pose
estimator. A trained pose head can replace it without changing the pipeline
contract.

## 4. Dataset contracts

### Identity dataset

`NameLabeledFaceDataset` expects a directory containing identity folders. It:

* sorts paths deterministically;
* normalizes condition suffixes such as `_wearing_mask` so an occlusion is not
  treated as a new person;
* returns `(image, identity_index, True)`;
* raises for missing, empty, corrupt, or invalid datasets.

Synthetic samples are available only through the explicit
`allow_synthetic_fixtures=True` test option. They are never the default.

WIDER FACE is a detection dataset and is not used as an identity-classification
dataset unless a separate identity manifest is supplied.

### Unlabeled dataset

`UnlabeledWildDataset` returns `(image, -1, False)`. It also fails fast by
default. It is used by the optional graph-adaptation phase and never receives
random identity labels.

### Attribute dataset

`CelebAAttributeDataset` uses only images that have real CelebA attribute
annotations. It does not append wild images with random attribute targets and
does not fabricate occlusion rectangles as segmentation ground truth.

Attribute prediction, occlusion quality, detection, and identity recognition
are separate tasks. They must not share a target tensor merely because the
image source is the same.

## 5. Recognition backbone and losses

### IResNet-100

The default `iresnet100` backbone uses the improved residual block layout
`[3, 13, 30, 3]`, produces a 512-dimensional embedding, applies BatchNorm, and
L2-normalizes the output. This is an implementation of the IResNet-shaped
architecture; official InsightFace weights are not assumed to exist. A trained
checkpoint is required for meaningful recognition.

### ViT-style option

`vit_face_base` is a generic 112x112 ViT with 8x8 patches, 196 patch tokens, a
CLS token, 12 transformer blocks, eight heads, and a normalized 512-dimensional
output. It should not be called TransFace or a pretrained FaceViT unless a
compatible pretrained checkpoint and training recipe are supplied.

### CurricularFace

`CurricularFaceLoss` and the classifier inside
`BroadFaceCurricularLoss` implement the angular margin and EMA curriculum
parameter `t`. Labels are validated instead of clamped.

### Memory queue

`BroadFaceCurricularLoss` maintains detached feature, label, and proxy queues.
The current batch contributes once to the current loss. Queue samples update
the classifier without sending stale feature gradients into the backbone.
The queue is explicitly enqueued after the optimizer step and gathers samples
across DDP ranks when distributed training is active.

The queue contains past examples, not new identity classes. Its size is a
memory/training hyperparameter and must be reported with the experiment.

## 6. Optional preprocessing modules

### Attribute parser

`ANetAttributeParser` predicts the 40 CelebA attributes. The local path uses
landmarks when available and falls back to an upper-face patch when they are
not. CelebA does not provide every occlusion attribute described in older
project notes; therefore the parser is not treated as a mask detector unless a
separate annotated checkpoint is supplied.

### Foreground mask

`SAMFaceSegmentor` is retained as an optional custom mask module. It is not
described as a full Meta SAM implementation. It must be checkpointed before it
is enabled. The pipeline applies a residual soft mask (`0.35 + 0.65 * mask`)
so uncertain masks cannot erase the entire face.

### Frontalization and super-resolution

`PIMFrontalizationGAN` and `D2SCGANSuperRes` are optional generators. They are
not run by default because an untrained generator can damage identity features.
They are enabled only when a compatible checkpoint is loaded and the landmark
pose signal exceeds the configured gate. Their quality should be validated by
an ablation against the aligned crop.

## 7. Dictionary-conditioned open-set recognition

`models/ddrc_solver.py` now uses the actual enrolled dictionary in every
unrolled sparse-coding step:

```text
minimize ||f - D x - e||² + lambda_x ||x||₁ + lambda_e ||e||₁
```

The solver updates coefficients using `D` and `Dᵀ`, estimates a sparse error
vector, and computes class residuals from class-specific dictionary atoms:

```text
r_k = mean((f - e - D_k x_k)²)
```

The classifier accepts either class prototypes or a multi-sample gallery. The
gallery is built from embeddings generated by the same preprocessing path as
queries; Phase 1 classifier weights are not treated as enrolled identities.

The classifier returns:

```python
{
    "identity": name or "unknown",
    "recognition_score": float,
    "residuals": per-class residuals,
}
```

Residual and margin thresholds are decision thresholds, not calibration by
themselves. A production deployment must select them using held-out known and
unknown identities. A one-class gallery has explicit semantics and no longer
crashes while reading a nonexistent second residual.

## 8. Training phases

### Phase 1 — identity backbone

`train.py` trains only on real identity-labelled face crops:

```text
identity dataset
  -> backbone
  -> BroadFaceCurricularLoss
  -> optimizer step
  -> queue enqueue
  -> checkpoint
```

Untrained detector, mask, and frontalizer modules are not inserted into this
path. The checkpoint stores:

* backbone state;
* classifier/curriculum/queue state;
* optimizer, scheduler, and AMP scaler state;
* model and loss configuration;
* identity label map;
* epoch and schema version.

### Phase 2 — attributes

Phase 2 is optional and runs only when `celeba_dir` and a valid CelebA
attribute file are configured. It trains the attribute logits on real CelebA
labels. It does not claim to train SAM or Wing Loss landmarks without those
targets. The saved artifact is `phase2_anet_attributes.pt`.

### Phase 3 — global unlabeled adaptation

Phase 3 is optional. It:

1. extracts embeddings for the complete unlabeled stream;
2. trains the graph edge predictor on labelled identity pairs;
3. creates globally indexed pseudo-labels rather than resetting IDs per batch;
4. uses sigmoid probabilities for confidence thresholds;
5. uses a blockwise nonnegative graph for large streams;
6. fine-tunes on labelled and pseudo-labelled samples using
   `L_labeled + lambda_semi * L_pseudo`;
7. saves both the GCN artifact and `phase3_adapted_backbone.pt`.

Large-scale deployments should replace the blockwise reference graph with a
FAISS or distributed nearest-neighbor implementation. The reference phase
does not materialize an `N x N x hidden` pair tensor for large collections.

## 9. Full execution architecture: training and inference

This section is the complete runtime flow for the current codebase. It separates
the offline training artifacts from the online recognition path and identifies
the main Python entry points involved at each stage.

### 9.1 Training entry point and initialization

The training command is:

```bash
python train.py \
  --data_dir <identity_face_crop_directory> \
  --checkpoint_dir <checkpoint_directory> \
  --backbone iresnet100
```

`train.py:main()` performs the following initialization:

```text
CLI arguments
  -> SystemConfig
  -> explicit dataset/checkpoint/backbone configuration
  -> DDP setup when torchrun variables and CUDA are available
  -> deterministic random seed per rank
  -> Phase 1 identity training
  -> optional Phase 2 attribute training
  -> optional Phase 3 graph adaptation
```

Missing required paths are not replaced with random data. The identity stream
must contain real labelled face images arranged in identity folders. The
unlabeled and CelebA streams are optional and are skipped explicitly when their
paths are not configured.

### 9.2 Phase 1 — identity backbone training

The implementation is `train_phase1_backbone_curricular()` in `train.py`.

#### Dataset construction

```text
cfg.dataset.train_data_dir
  -> NameLabeledFaceDataset
  -> deterministic image paths and canonical identity names
  -> image resize/normalization to 112x112 and optional train flip
  -> (images, identity_labels, is_labeled=True)
```

Identity labels are validated against the classifier class count. WIDER FACE
bounding-box images are not silently converted into identity labels.

#### Model construction

```text
get_backbone(cfg)
  -> IResNet-100 or generic ViT-Face backbone
  -> normalized embedding f in R^512

BroadFaceCurricularLoss
  -> classifier weight matrix
  -> CurricularFace angular-margin calculation
  -> EMA curriculum parameter t
  -> fixed-capacity detached feature/proxy queue

RecognitionModel
  -> wraps backbone and loss together for DDP synchronization
```

The detector, optional foreground module, and frontalizer are intentionally not
inserted here. Adding an untrained preprocessing network to identity training
would make training and inference distributions inconsistent.

#### Per-batch execution

For every batch, the runtime flow is:

```text
images, labels
  -> move to CPU/GPU training device
  -> backbone(images)
  -> normalized embeddings
  -> current-batch CurricularFace loss
  -> compensated detached queue loss, if the queue is populated
  -> total loss
  -> backward pass with optional CUDA AMP
  -> optimizer.step()
  -> queue enqueue after the optimizer update
```

The current batch contributes once to the current loss. Queue features are
detached so stale embeddings update the classifier without sending stale
representation gradients through the backbone. In DDP, embeddings and labels
are gathered before being added to each rank's identical queue.

#### Phase 1 checkpoint flow

At periodic intervals and at the end of training, the following are saved:

```text
recognition_backbone.pt
latest_backbone.pt
phase1_epoch_<N>.pt
```

Each new checkpoint stores:

```text
schema_version
kind = recognition_backbone
epoch
backbone_state
broadface_loss_fn state, including EMA and queue
optimizer state
scheduler state
AMP scaler state
model/loss configuration
input size
class-to-index map
```

The `latest_backbone.pt` file is used for a true resume, while
`recognition_backbone.pt` is the completed Phase 1 artifact used by inference.

### 9.3 Phase 2 — annotated attribute training

The optional entry point is `train_phase2_anet_attributes()` in `train.py`.

```text
celeba_dir + list_attr_celeba annotation
  -> CelebAAttributeDataset
  -> real 40-dimensional attribute targets
  -> ANetAttributeParser
  -> BCEWithLogitsLoss
  -> Adam update
  -> phase2_anet_attributes.pt
```

Only real CelebA attribute rows are used. Wild images without attribute labels
are excluded. This phase does not claim to train facial landmarks, SAM masks,
or PIM without separate annotated data and training routines.

When an attribute checkpoint is loaded for inference:

```text
aligned crop + detector landmarks
  -> landmark-aware ANet local branch
  -> 40 attribute logits
  -> optional occlusion-quality indicator
```

The attribute parser is never silently enabled with random weights. A caller
requesting an optional attribute/mask/frontalization branch without a matching
checkpoint receives an explicit error.

### 9.4 Phase 3 — global unlabeled graph adaptation

The optional entry point is `train_phase3_semi_supervised_gcn()` in `train.py`.

```text
unlabeled face-crop dataset
  -> frozen Phase 1 backbone
  -> one embedding per globally indexed sample
  -> GCN edge predictor trained on labelled identity pairs
  -> global pseudo-label graph
  -> confidence filtering and globally unique cluster IDs
  -> labelled + pseudo-labelled backbone fine-tuning
  -> phase3_semi_gcn_predictor.pt
  -> phase3_adapted_backbone.pt
```

For small graphs, the trained GCN edge head predicts pair probabilities. For
large graphs, `generate_global_pseudo_labels()` constructs a blockwise,
nonnegative nearest-neighbour graph to avoid an `N x N x hidden` allocation.
Pseudo-label IDs are assigned once for the complete graph; they do not restart
at each DataLoader batch.

Fine-tuning uses:

```text
L_total = L_labelled + lambda_semi * L_pseudo
```

The adapted backbone is saved separately so the original supervised checkpoint
remains reproducible.

### 9.5 Checkpoint loading before inference

`OccuPoseBroadDictPipeline.load_pretrained_weights()` searches for the newest
compatible backbone artifact in this order:

```text
phase3_adapted_backbone.pt
recognition_backbone.pt
latest_backbone.pt
phase1_backbone_curricular.pt
phase1_epoch_100.pt
```

The loader checks the checkpoint schema/kind and uses the stored architecture
metadata when rebuilding the backbone. It removes only a known DDP
`module.` prefix and rejects unsupported checkpoint metadata instead of
silently continuing with incompatible state.

The optional Phase 2 and Phase 3 artifacts are then loaded separately:

```text
phase2_anet_attributes.pt
  -> optional attribute parser and any explicitly stored preprocessing state

phase3_semi_gcn_predictor.pt
  -> optional GCN adapter
```

Loading a backbone does not create an identity gallery. Enrollment is a separate
required operation.

### 9.6 Gallery enrollment flow

`OccuPoseBroadDictPipeline.enroll_dataset()` builds the open-set gallery:

```text
identity image path
  -> RGB load
  -> shared 112x112 evaluation transform
  -> shared crop/alignment/preprocessing path
  -> recognition backbone
  -> normalized embedding
  -> identity-specific gallery atoms
  -> DDRCClassifier.enroll_gallery()
```

The gallery uses real identity images and their canonical names. Phase 1
classifier weights are not treated as enrollment samples. Multiple samples per
identity can be retained as multiple dictionary atoms, allowing the open-set
solver to model within-person variation.

### 9.7 Online inference flow

The normal CLI entry point is `predict_occluded_faces.py`. Its per-image flow
is:

```text
image file
  -> PIL RGB load
  -> deterministic resize to model scene resolution
  -> normalization to [-1, 1]
  -> OccuPoseBroadDictPipeline.forward()
```

Inside `forward()`:

```text
1. Detector
   image tensor
     -> YuNet and/or MTCNN
     -> candidate boxes, scores, five-point landmarks
     -> index-preserving Gaussian Soft-NMS

2. Empty-detection gate
   no boxes -> return []
   no fabricated full-image detection is created

3. Per-face crop
   box coordinates
     -> clamp to image bounds
     -> reject crops smaller than the minimum valid size
     -> crop face patch
     -> convert landmarks from image coordinates to crop coordinates

4. Alignment
   crop + local five-point landmarks
     -> similarity transform to canonical 112x112 coordinates
     -> box-only resize fallback when landmarks are invalid
     -> aligned quality flag

5. Optional preprocessing
   aligned crop
     -> checkpointed soft foreground mask, if enabled
     -> landmark-aware attributes, if available
     -> landmark-derived yaw/pitch/roll quality signal
     -> checkpointed frontalizer only when the pose gate is exceeded

6. Recognition embedding
   processed 112x112 crop
     -> configured backbone
     -> L2-normalized embedding f

7. Open-set decision
   f + enrolled dictionary
     -> sparse coefficients x and error e
     -> class-specific residuals r_k
     -> residual and margin gates
     -> identity name or 'unknown'

8. Result assembly
   detector score, recognition score, pose, alignment state, residuals,
   attributes, and original model-space box
```

The result is sorted by combined display confidence, but the raw detector and
recognition scores remain separate in the output contract. The prediction CLI
maps the model-space box from the resized scene back to the original image
coordinates before drawing annotations and writing CSV output.

### 9.8 Inference failure and safety paths

```text
missing detector backend       -> empty detections -> []
missing gallery                -> detected faces classified as unknown
missing optional checkpoint   -> optional branch disabled or explicit error
invalid crop                  -> face skipped, other faces continue
invalid image                 -> clear input/decode error
missing WIDER ground truth    -> evaluation aborts, no AP is emitted
```

These paths are intentional: a failed optional module must not turn into a
random identity, a fabricated face, or a misleading benchmark score.

## 10. Inference contract

`OccuPoseBroadDictPipeline.forward()` accepts a normalized tensor with shape
`(1, 3, H, W)` and returns one result per detected face. Each result contains:

```python
{
    "box": [x1, y1, x2, y2],
    "box_xyxy": [x1, y1, x2, y2],
    "identity": "person name" or "unknown",
    "is_unknown": bool,
    "detector_score": float,
    "recognition_score": float,
    "confidence": detector_score * recognition_score,
    "is_occluded": bool,
    "attributes": dict,
    "pose": {"yaw", "pitch", "roll", "valid"},
    "aligned": bool,
    "frontalized": bool,
    "residuals": list,
}
```

If the detector returns no boxes, the pipeline returns `[]`. It does not create
a full-image result. The prediction CLI maps model-space boxes back to original
image coordinates for annotation.

Gallery enrollment must happen before identity classification. Without a
gallery, valid detections are returned as `unknown` rather than assigned to a
random identity.

## 11. Checkpoint compatibility

All new training artifacts use `schema_version=2`. A checkpoint loader validates
the required state and removes only an expected DDP `module.` prefix. It does
not silently continue when a required backbone, parser, or GCN state is absent.

The model bundle should record:

```text
architecture
embedding_dim
input size
normalization
enabled preprocessing modules
preprocessing thresholds
label map
data manifest hash
training commit
evaluation metrics
```

The following files are the principal artifacts:

```text
checkpoints/recognition_backbone.pt
checkpoints/latest_backbone.pt
checkpoints/phase2_anet_attributes.pt
checkpoints/phase3_semi_gcn_predictor.pt
checkpoints/phase3_adapted_backbone.pt
```

## 12. WIDER FACE evaluation

`eval_widerface.py` is a ground-truth-required evaluator. It:

* discovers `wider_face_split/wider_face_val_bbx_gt.txt`;
* preserves event-relative image paths;
* rejects malformed, duplicate, truncated, or partial annotations;
* assigns difficulty using matched ground-truth face height;
* includes unmatched predictions as false positives;
* never reports confidence averages as AP or mAP.

If ground truth is absent, evaluation stops with an explicit error. The existing
`results/widerface_eval/widerface_industry_metrics.csv` was produced before
this safeguard and contains `GroundTruth_Loaded=False`; it must not be used as
a WIDER FACE AP claim.

## 13. Source mapping and limits

The implementation is informed by the repository's extracted papers:

* CurricularFace: angular margin with an EMA curriculum parameter.
* BroadFace: detached past embeddings, identity proxies, and compensation.
* Deep Learning Face Attributes in the Wild: identity-pretrained attribute
  representation and local facial regions.
* DDRC: gallery and auxiliary dictionary reasoning in feature space.
* PIM/D2SC-GAN: pose/frontalization and shallow/deep restoration ideas.
* Unlabeled face clustering: global graph construction and uncertainty-aware
  pseudo-label adaptation.

The code does not claim that every original paper is reproduced line-for-line.
In particular, external pretrained weights, official SAM image encoders,
research-specific GAN discriminators, and official GCN clustering datasets are
not assumed to be present. Optional modules are therefore explicit and
checkpoint-gated rather than silently used in a random state.

## 14. Verification commands

From the repository root:

```bash
python -m compileall -q .
python -m unittest tests.test_regressions -v
python -m unittest discover -s tests -p 'test*.py'
python eval_widerface.py --image_dir WIDER_val/images --gt_file wider_face_split/wider_face_val_bbx_gt.txt
```

The first two checks do not require model weights. Full training, inference,
and model smoke tests require the configured PyTorch environment and real
detector/backbone checkpoints.
