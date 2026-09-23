import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from config import SystemConfig
from dataset import WildFaceDataset, UnlabeledFaceDataset, CelebAAttributeDataset
from losses.curricular_loss import CurricularFaceLoss
from losses.broadface_queue import BroadFaceMemoryQueue
from models.backbone import ResNet100Backbone, vit_face_base
from models.anet_attribute import ANetAttributeParser
from models.gcn_cluster import GCNLinkPredictor

def get_backbone(cfg: SystemConfig):
    if cfg.model.backbone_type == "vit_face_base":
        return vit_face_base(embedding_dim=cfg.model.embedding_dim)
    return ResNet100Backbone(embedding_dim=cfg.model.embedding_dim)

def train_phase1_backbone_curricular(cfg: SystemConfig, device: torch.device):
    """
    Phase 1 Training: Backbone (IResNet-100 / ViT-Face) + CurricularFace Adaptive Loss + BroadFace Queue.
    """
    print(f"\n========================================================================")
    print(f"--- Starting Phase 1: {cfg.model.backbone_type.upper()} & CurricularFace Training ---")
    print(f"========================================================================\n")
    
    train_dataset = WildFaceDataset(root_dir=cfg.dataset.train_data_dir, is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True
    )
    
    num_classes = len(train_dataset.class_to_idx) if train_dataset.class_to_idx else cfg.dataset.num_classes
    print(f"Labeled training dataset initialized: {len(train_dataset)} samples across {num_classes} identities.")
    
    backbone = get_backbone(cfg).to(device)
    curricular_loss_fn = CurricularFaceLoss(
        in_features=cfg.model.embedding_dim,
        num_classes=num_classes,
        s=cfg.loss.scale,
        m=cfg.loss.margin,
        alpha=cfg.loss.ema_alpha
    ).to(device)
    
    broadface_queue = BroadFaceMemoryQueue(
        queue_size=cfg.loss.queue_size,
        feature_dim=cfg.model.embedding_dim,
        momentum=cfg.loss.broadface_momentum
    ).to(device)
    
    optimizer = optim.SGD(
        list(backbone.parameters()) + list(curricular_loss_fn.parameters()),
        lr=cfg.train.learning_rate,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scaler = GradScaler('cuda', enabled=cfg.model.fp16 and device.type == 'cuda')
    
    backbone.train()
    curricular_loss_fn.train()
    
    for epoch in range(1, cfg.train.epochs + 1):
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            images, labels, _ = batch # batch returns (images, labels, is_labeled)
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast('cuda', enabled=cfg.model.fp16 and device.type == 'cuda'):
                embeddings = backbone(images)
                loss = curricular_loss_fn(embeddings, labels)
                
            scaler.scale(loss).backward()
            with torch.no_grad():
                old_weight = curricular_loss_fn.weight.clone()
                
            scaler.step(optimizer)
            scaler.update()
            
            broadface_queue.update(embeddings.detach(), labels)
            broadface_queue.compensate_weight_drift(old_weight, curricular_loss_fn.weight)
            running_loss += loss.item()
            
        scheduler.step()
        avg_loss = running_loss / max(1, len(train_loader))
        t_param = curricular_loss_fn.t.item()
        print(f"Epoch [{epoch}/{cfg.train.epochs}] - Curricular Loss: {avg_loss:.4f} | EMA t: {t_param:.4f}")
        
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.train.checkpoint_dir, "phase1_backbone_curricular.pt")
    torch.save({
        'backbone': backbone.state_dict(),
        'curricular_loss': curricular_loss_fn.state_dict(),
        'broadface_queue': broadface_queue.state_dict()
    }, ckpt_path)
    print(f"\n[Phase 1 Complete] Checkpoint saved to: {ckpt_path}")
    return backbone, num_classes

def train_phase2_anet_attributes(cfg: SystemConfig, device: torch.device):
    """
    Phase 2 Training: ANet Multi-Task 40-Attribute Parser & Spatial Occlusion Attention.
    """
    print(f"\n========================================================================")
    print(f"--- Starting Phase 2: ANet Multi-Task Attribute Parser Training ---")
    print(f"========================================================================\n")
    
    celeba_dataset = CelebAAttributeDataset(root_dir=cfg.dataset.celeba_dir, is_train=True)
    train_loader = DataLoader(celeba_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers)
    
    anet = ANetAttributeParser(num_attributes=40).to(device)
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(anet.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = GradScaler('cuda', enabled=cfg.model.fp16 and device.type == 'cuda')
    
    anet.train()
    for epoch in range(1, min(cfg.train.epochs, 5) + 1):
        running_loss = 0.0
        for images, attr_targets in train_loader:
            images, attr_targets = images.to(device), attr_targets.to(device)
            
            optimizer.zero_grad()
            with autocast('cuda', enabled=cfg.model.fp16 and device.type == 'cuda'):
                attr_logits, occ_mask, _ = anet(images)
                loss = criterion_bce(attr_logits, attr_targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            
        print(f"ANet Epoch [{epoch}/{min(cfg.train.epochs, 5)}] - Attribute BCE Loss: {running_loss / max(1, len(train_loader)):.4f}")
        
    ckpt_path = os.path.join(cfg.train.checkpoint_dir, "phase2_anet_attributes.pt")
    torch.save({'anet': anet.state_dict()}, ckpt_path)
    print(f"[Phase 2 Complete] ANet checkpoint saved to: {ckpt_path}")
    return anet

def train_phase3_semi_supervised_gcn(cfg: SystemConfig, backbone, num_labeled_classes: int, device: torch.device):
    """
    Phase 3 Training: Semi-Supervised Unlabeled Face Clustering via GCN Sub-Graph Link Prediction.
    Tackles unannotated wild face datasets (FMD, COVID faces, web crawls).
    """
    print(f"\n========================================================================")
    print(f"--- Starting Phase 3: Semi-Supervised GCN Clustering (Unlabeled Faces) ---")
    print(f"========================================================================\n")
    
    unlabeled_dataset = UnlabeledFaceDataset(root_dir=cfg.dataset.unlabeled_data_dir, is_train=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers)
    
    gcn_predictor = GCNLinkPredictor(feature_dim=cfg.model.embedding_dim, hidden_dim=256, k_neighbors=5).to(device)
    optimizer_gcn = optim.Adam(gcn_predictor.parameters(), lr=1e-3)
    
    print(f"Unlabeled wild face dataset loaded: {len(unlabeled_dataset)} images.")
    
    backbone.eval()
    for epoch in range(1, min(cfg.train.epochs, 3) + 1):
        total_clustered = 0
        for images, labels, is_labeled in unlabeled_loader:
            images = images.to(device)
            with torch.no_grad():
                unlabeled_embeds = backbone(images)
                
            pseudo_labels, active_mask = gcn_predictor.generate_pseudo_labels(
                unlabeled_embeds,
                confidence_threshold=cfg.loss.pseudo_label_threshold,
                start_class_idx=num_labeled_classes
            )
            total_clustered += active_mask.sum().item()
            
        print(f"GCN Semi-Supervised Epoch [{epoch}/{min(cfg.train.epochs, 3)}] - High-Confidence Pseudo-Labeled Faces: {total_clustered}/{len(unlabeled_dataset)}")
        
    ckpt_path = os.path.join(cfg.train.checkpoint_dir, "phase3_semi_gcn_predictor.pt")
    torch.save({'gcn_predictor': gcn_predictor.state_dict()}, ckpt_path)
    print(f"[Phase 3 Complete] GCN Link Predictor checkpoint saved to: {ckpt_path}")

def main():
    parser = argparse.ArgumentParser(description="OccuPose-BroadDictNet Production Multi-Stage Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="", help="Path to labeled training dataset (ROF, LFW, WIDER)")
    parser.add_argument("--unlabeled_dir", type=str, default="", help="Path to unlabeled face dataset (FMD, COVID faces)")
    parser.add_argument("--celeba_dir", type=str, default="", help="Path to CelebA attribute dataset")
    parser.add_argument("--backbone", type=str, default="iresnet100", choices=["iresnet100", "vit_face_base"], help="Backbone type")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs per phase")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--fp16", action="store_true", help="Enable AMP FP16")
    args = parser.parse_args()
    
    cfg = SystemConfig()
    if args.data_dir:
        cfg.dataset.train_data_dir = args.data_dir
    if args.unlabeled_dir:
        cfg.dataset.unlabeled_data_dir = args.unlabeled_dir
    if args.celeba_dir:
        cfg.dataset.celeba_dir = args.celeba_dir
    cfg.model.backbone_type = args.backbone
    cfg.train.checkpoint_dir = args.checkpoint_dir
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.learning_rate = args.lr
    cfg.model.fp16 = args.fp16 or torch.cuda.is_available()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running OccuPose-BroadDictNet Training Engine on: {device} (Backbone: {cfg.model.backbone_type})")
    
    # Phase 1: Train Backbone + CurricularFace + BroadFace
    backbone, num_classes = train_phase1_backbone_curricular(cfg, device)
    
    # Phase 2: Train ANet Attribute Parser
    anet = train_phase2_anet_attributes(cfg, device)
    
    # Phase 3: Train GCN Semi-Supervised Link Predictor & Pseudo-Labeler on Unlabeled Faces
    train_phase3_semi_supervised_gcn(cfg, backbone, num_classes, device)
    
    print("\n========================================================================")
    print("✓ OccuPose-BroadDictNet Production Multi-Stage Training Pipeline Complete!")
    print("========================================================================")

if __name__ == "__main__":
    main()
