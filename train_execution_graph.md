# `train.py` Runtime Execution Graph & Function Call Dependencies

> **System Overview**: This document details the exact line-by-line runtime execution graph, function calls, module instantiations, and file interactions during multi-stage training in `train.py`.

---

## 1. End-to-End Runtime Execution Flowchart

```mermaid
flowchart TD
    CLI[CLI Launch: python train.py] --> Main[main: Parse Arguments & Initialize SystemConfig]
    Main --> DDPSetup[setup_ddp: Initialize Distributed NCCL / GLOO & CUDA Devices]
    
    subgraph Phase 1: Identity Feature Embedding & BroadCurricular Training
        DDPSetup --> P1_Init[WildFaceDataset Loader & DataLoader]
        P1_Init --> P1_Backbone[get_backbone: IResNet100 / FasterRCNNBackbone / ViT-Face]
        P1_Backbone --> P1_Models[SAMFaceSegmentor & PIMFrontalizationGAN & FasterRCNNFaceDetector]
        P1_Models --> P1_LossInit[BroadFaceCurricularLoss: CurricularFace + BroadFace Queue N_q=32,768]
        P1_LossInit --> P1_Loop[Phase 1 Batch Training Loop: Epochs 1..100]
        P1_Loop --> P1_SAM[sam_segmentor: Extract Foreground Mask M_SAM]
        P1_SAM --> P1_Pose{abs estimated_yaw > 20.0?}
        P1_Pose -- Yes --> P1_PIM[PIMFrontalizationGAN: Frontalize Crop]
        P1_Pose -- No --> P1_Embedding[Backbone Forward Pass: Extract 512-d Feature f]
        P1_PIM --> P1_Embedding
        P1_Embedding --> P1_LossEval[BroadFaceCurricularLoss: Compute Margin & Queue Update]
        P1_LossEval --> P1_Backprop[GradScaler AMP FP16 + SGD Optimizer Step]
        P1_Backprop --> P1_Ckpt[Save Checkpoints: phase1_epoch_X.pt & phase1_backbone_curricular.pt]
    end

    subgraph Phase 2: Multi-Task Attribute, SAM & Occlusion Training
        P1_Ckpt --> P2_Init[CelebAAttributeDataset Loader & DataLoader]
        P2_Init --> P2_ANet[ANetAttributeParser: init_from_backbone Transfer Learning]
        P2_ANet --> P2_SAM[SAMFaceSegmentor & PIMFrontalizationGAN & WingLoss]
        P2_SAM --> P2_Loop[Phase 2 Batch Training Loop: Epochs 1..30]
        P2_Loop --> P2_Pose{abs estimated_yaw > 20.0?}
        P2_Pose -- Yes --> P2_PIM[PIMFrontalizationGAN: Frontalize Crop]
        P2_Pose -- No --> P2_ANetForward[ANet Forward: 40 Attribute Logits + 7x7 Spatial Mask M_spatial]
        P2_PIM --> P2_ANetForward
        P2_ANetForward --> P2_SAMForward[sam_segmentor: Predict SAM Mask M_SAM]
        P2_SAMForward --> P2_JointLoss[Compute Joint Multi-Task Loss: BCE Attr + BCE Mask + BCE SAM + WingLoss]
        P2_JointLoss --> P2_Backprop[GradScaler AMP FP16 + Adam Optimizer Step]
        P2_Backprop --> P2_Ckpt[Save Checkpoints: phase2_epoch_X.pt & phase2_anet_attributes.pt]
    end

    subgraph Phase 3: Semi-Supervised GCN Graph Clustering
        P2_Ckpt --> P3_Init[UnlabeledFaceDataset Loader & DataLoader: FMD / COVID Faces]
        P3_Init --> P3_GCN[GCNLinkPredictor: k-NN Graph & 2-Layer GraphConvBlock]
        P3_GCN --> P3_Loop[Phase 3 Batch Training Loop: Epochs 1..10]
        P3_Loop --> P3_Extract[Backbone Forward Pass: Extract Unlabeled Embeddings]
        P3_Extract --> P3_Pseudo[GCNLinkPredictor.generate_pseudo_labels: BFS Connected Components]
        P3_Pseudo --> P3_Ckpt[Save Checkpoint: phase3_semi_gcn_predictor.pt]
    end

    P3_Ckpt --> Complete[Training Pipeline Complete & Weights Synced]
```

---

## 2. Phase-by-Phase Function Call & File Interaction Breakdown

### **Phase 1: Backbone & Curricular Loss Training (`train_phase1_backbone_curricular`)**

```
train.py (Line 44)
 │
 ├──► dataset.py: WildFaceDataset(root_dir, is_train=True)          [Line 59]
 │      └─► PurePyTorchImageTransform / get_default_transform        [dataset.py: Line 39]
 │
 ├──► models/backbone.py: get_backbone(cfg)                         [Line 83]
 │      ├─► ResNet100Backbone (IResNet-100 stages [3, 13, 30, 3])    [backbone.py: Line 272]
 │      ├─► FasterRCNNBackbone (ResNet-50 FPN intermediate features) [backbone.py: Line 295]
 │      └─► vit_face_base (FaceVisionTransformer 196 patch tokens)   [backbone.py: Line 279]
 │
 ├──► models/detector.py: FasterRCNNFaceDetector(weights_dir)       [Line 84]
 ├──► models/sam_segmentor.py: SAMFaceSegmentor()                    [Line 86]
 ├──► models/pim_frontalizer.py: PIMFrontalizationGAN()             [Line 88]
 │
 ├──► losses/broadface_queue.py: BroadFaceCurricularLoss()          [Line 91]
 │      ├─► losses/curricular_loss.py: CurricularFaceLoss (EMA t)   [curricular_loss.py: Line 7]
 │      └─► BroadFace Memory Queue N_q = 32,768 (Drift Compensated)  [broadface_queue.py: Line 78]
 │
 └──► Training Loop (Lines 152–195):
        ├─► sam_segmentor(images) ──► SAM Mask M_SAM                [Line 176]
        ├─► frontalizer(segmented_images, estimated_yaw)            [Line 184]
        ├─► backbone(segmented_images) ──► 512-d Feature Vector f    [Line 186]
        ├─► broadface_loss_fn(embeddings, labels)                   [Line 187]
        └─► torch.save() ──► checkpoints/phase1_epoch_X.pt         [Line 187]
```

---

### **Phase 2: ANet Attribute & SAM Mask Training (`train_phase2_anet_attributes`)**

```
train.py (Line 224)
 │
 ├──► dataset.py: CelebAAttributeDataset(root_dir, is_train=True)    [Line 234]
 ├──► models/anet_attribute.py: ANetAttributeParser(num_attributes=40) [Line 245]
 │      └─► anet.init_from_backbone(backbone)                       [Line 266]
 ├──► models/sam_segmentor.py: SAMFaceSegmentor()                    [Line 246]
 ├──► models/pim_frontalizer.py: PIMFrontalizationGAN()             [Line 247]
 │
 ├──► Loss Criteria (Lines 271–273):
 │      ├─► nn.BCEWithLogitsLoss() ──► 40 Attribute Logits Loss     [Line 271]
 │      ├─► nn.BCELoss() ──► 7x7 Spatial Occlusion Map Loss         [Line 272]
 │      └─► losses/wing_loss.py: WingLoss(w=10.0, epsilon=2.0)     [Line 273]
 │
 └──► Training Loop (Lines 281–315):
        ├─► frontalizer(images, estimated_yaw) if |yaw| > 20.0       [Line 295]
        ├─► anet(images) ──► attr_logits, occ_mask, is_occluded      [Line 299]
        ├─► sam_segmentor(images) ──► sam_mask                      [Line 300]
        ├─► Multi-Task Joint Loss Calculation                        [Line 309]
        └─► torch.save() ──► checkpoints/phase2_epoch_X.pt         [Line 326]
```

---

### **Phase 3: Semi-Supervised GCN Clustering (`train_phase3_semi_supervised_gcn`)**

```
train.py (Line 336)
 │
 ├──► dataset.py: UnlabeledFaceDataset(root_dir, is_train=True)     [Line 346]
 ├──► models/gcn_cluster.py: GCNLinkPredictor(feature_dim=512)     [Line 354]
 │      └─► 2-Layer GraphConvBlock + Pairwise Edge MLP             [gcn_cluster.py: Line 15]
 │
 └──► Training Loop (Lines 366–393):
        ├─► backbone(unlabeled_images) ──► unlabeled_embeds         [Line 381]
        ├─► gcn_predictor.generate_pseudo_labels(unlabeled_embeds) [Line 350]
        │      └─► BFS Connected Component Pseudo-Labeler           [gcn_cluster.py: Line 85]
        └─► torch.save() ──► checkpoints/phase3_semi_gcn_predictor.pt [Line 397]
```

---

## 3. Line-by-Line Code & File Interaction Table

| Line Number Range | Function / Operation | Module / File Called | Output Artifact / Action |
| :--- | :--- | :--- | :--- |
| **Lines 14–23** | Module Imports | `config.py`, `dataset.py`, `losses/`, `models/` | Imports configurations, datasets, losses, and models |
| **Lines 25–30** | `get_backbone(cfg)` | `models/backbone.py` | Instantiates `IResNet-100`, `FasterRCNNBackbone`, or `vit_face_base` |
| **Lines 32–47** | `setup_ddp()` | `torch.distributed` | Configures multi-GPU PyTorch NCCL/GLOO DDP ranks |
| **Lines 59–72** | Phase 1 Data Load | `dataset.py` (`WildFaceDataset`) | Loads labeled face images and creates PyTorch `DataLoader` |
| **Lines 83–89** | Phase 1 Models Init | `models/backbone.py`, `models/detector.py`, `models/sam_segmentor.py`, `models/pim_frontalizer.py` | Instantiates `backbone`, `FasterRCNNFaceDetector`, `SAMFaceSegmentor`, and `PIMFrontalizationGAN` |
| **Lines 91–99** | Loss Function Init | `losses/broadface_queue.py` | Instantiates `BroadFaceCurricularLoss` ($N_q = 32,768$) |
| **Lines 176–187** | Phase 1 Step | `sam_segmentor`, `frontalizer`, `backbone`, `broadface_loss_fn` | Processes batch: SAM mask $\to$ Pose Frontalize if yaw $>20^\circ$ $\to$ Backbone Embedding $\to$ Loss |
| **Lines 189–191** | Phase 1 Opt Step | `torch.amp.GradScaler`, `torch.optim.SGD` | Scaled backward pass, optimizer step, AMP FP16 update |
| **Lines 234–240** | Phase 2 Data Load | `dataset.py` (`CelebAAttributeDataset`) | Loads CelebA 40-attribute images and creates `DataLoader` |
| **Lines 245–248** | Phase 2 Models Init | `models/anet_attribute.py`, `models/sam_segmentor.py`, `models/pim_frontalizer.py` | Instantiates `ANetAttributeParser`, `SAMFaceSegmentor`, `PIMFrontalizationGAN` |
| **Lines 266** | Transfer Learning | `models/anet_attribute.py` (`init_from_backbone`) | Transfers Phase 1 backbone conv weights into ANet feature extractor |
| **Lines 271–273** | Phase 2 Loss Init | `losses/wing_loss.py`, `torch.nn` | Instantiates `BCEWithLogitsLoss`, `BCELoss`, and `WingLoss` |
| **Lines 290–310** | Phase 2 Step | `frontalizer`, `anet`, `sam_segmentor` | Evaluates pose yaw $\to$ ANet attributes & spatial map $\to$ SAM mask $\to$ Multi-Task Loss |
| **Lines 346–352** | Phase 3 Data Load | `dataset.py` (`UnlabeledFaceDataset`) | Loads unannotated wild face images (FMD, COVID faces) |
| **Lines 354** | GCN Predictor Init | `models/gcn_cluster.py` (`GCNLinkPredictor`) | Instantiates GCN 2-layer graph convolutional link predictor |
| **Lines 381–388** | Phase 3 Step | `backbone`, `models/gcn_cluster.py` | Extracts embeddings $\to$ BFS Connected Component Pseudo-Labeling |
| **Lines 400–440** | `main()` CLI Entry | `argparse`, `train_aqua.cmd` | Parses CLI arguments and runs Phase 1, Phase 2, and Phase 3 sequentially |
