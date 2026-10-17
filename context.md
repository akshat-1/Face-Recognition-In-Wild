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

## 9. Inference contract

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

## 10. Checkpoint compatibility

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

## 11. WIDER FACE evaluation

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

## 12. Source mapping and limits

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

## 13. Verification commands

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
