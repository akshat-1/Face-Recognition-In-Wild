# AI handoff — Face Recognition in the Wild

Read `context.md` before changing the project. It is the source of truth for
the implemented architecture and checkpoint contracts.

## Current executable design

```text
image -> trained AQUA detector (or YuNet/MTCNN fallback) -> indexed Soft-NMS
       -> five-point alignment -> optional checkpointed preprocessing
       -> IResNet-100 or generic ViT embedding
       -> enrolled dictionary/open-set classifier
       -> known identity or unknown
```

Training is task-separated:

1. Phase 0 trains a class-agnostic Faster R-CNN from AQUA `bb_label` YOLO
   boxes and saves `face_detector.pt`.
2. Phase 1 trains an identity backbone on real labelled face crops with
   CurricularFace and an explicit BroadFace-style detached memory queue.
3. Phase 2 is optional and trains the 40 CelebA attribute logits only on real
   CelebA annotations.
4. Phase 3 is optional and performs globally indexed unlabeled graph
   adaptation, then saves the adapted backbone.

## Non-negotiable safety rules

* Missing datasets fail fast; random fallback data is test-only.
* Empty detector output remains empty; no full-image face is fabricated.
* Box, score, and landmark rows stay aligned through Soft-NMS.
* Enrollment and query embeddings use the same preprocessing path.
* Random SAM/PIM/attribute modules are never enabled implicitly.
* Phase 1 classifier weights are not identity gallery prototypes.
* A confidence average is never reported as WIDER FACE AP.

## Important entry points

```text
train.py
    train_phase0_face_detector
    train_phase1_backbone_curricular
    train_phase2_anet_attributes
    train_phase3_semi_supervised_gcn

pipeline/alignment.py
    align_face_crop
    estimate_pose_from_landmarks

pipeline/wild_face_pipeline.py
    OccuPoseBroadDictPipeline.load_pretrained_weights
    OccuPoseBroadDictPipeline.enroll_dataset
    OccuPoseBroadDictPipeline.forward

models/detector.py
    SOTAFaceDetector
    FasterRCNNFaceDetector compatibility facade

models/ddrc_solver.py
    LISTASparseSolver
    DDRCClassifier

eval_widerface.py
    find_widerface_gt_file
    load_widerface_gt
    compute_widerface_metrics
```

## Checkpoint artifacts

```text
checkpoints/recognition_backbone.pt
checkpoints/face_detector.pt
checkpoints/latest_backbone.pt
checkpoints/phase2_anet_attributes.pt
checkpoints/phase3_semi_gcn_predictor.pt
checkpoints/phase3_adapted_backbone.pt
```

New artifacts use `schema_version=2`. Loaders must reject incompatible
architectures, missing required state, and incompatible preprocessing settings.

## Verification

From the repository root:

```bash
python -m compileall -q .
python -m unittest tests.test_regressions -v
python -m unittest discover -s tests -p 'test*.py'
```

The model tests require the configured PyTorch environment. The regression
evaluation tests are intentionally importable without PyTorch.
