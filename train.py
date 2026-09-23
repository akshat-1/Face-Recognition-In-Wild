import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from config import SystemConfig
from dataset import WildFaceDataset, CelebAAttributeDataset
from losses.curricular_loss import CurricularFaceLoss
from losses.broadface_queue import BroadFaceMemoryQueue
from models.backbone import ResNet100Backbone
from models.anet_attribute import ANetAttributeParser

def train_phase1_backbone_curricular(cfg: SystemConfig, device: torch.device):
    """
    Phase 1 Training: IResNet-100 Backbone + CurricularFace Adaptive Margin Loss + BroadFace Memory Queue.
    Supports AMP Automatic Mixed Precision (FP16) training for speed and memory efficiency.
    """
    print(f"\n========================================================================")
    print(f"--- Starting Phase 1: IResNet-100 & CurricularFace Training ---")
    print(f"========================================================================\n")
    
    # Dataset and DataLoader
    train_dataset = WildFaceDataset(root_dir=cfg.dataset.train_data_dir, is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True
    )
    
    num_classes = len(train_dataset.class_to_idx) if train_dataset.class_to_idx else cfg.dataset.num_classes
    print(f"Training dataset initialized: {len(train_dataset)} samples across {num_classes} identities.")
    
    # Models
    backbone = ResNet100Backbone(
        embedding_dim=cfg.model.embedding_dim
    ).to(device)
    
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
    
    # Optimizer, Scaler & Cosine Learning Rate Scheduler
    optimizer = optim.SGD(
        list(backbone.parameters()) + list(curricular_loss_fn.parameters()),
        lr=cfg.train.learning_rate,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scaler = GradScaler(enabled=cfg.model.fp16)
    
    backbone.train()
    curricular_loss_fn.train()
    
    for epoch in range(1, cfg.train.epochs + 1):
        running_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # AMP Mixed Precision Forward Pass
            with autocast(enabled=cfg.model.fp16):
                embeddings = backbone(images)
                loss = curricular_loss_fn(embeddings, labels)
                
            # Scaled Backward Pass
            scaler.scale(loss).backward()
            
            with torch.no_grad():
                old_weight = curricular_loss_fn.weight.clone()
                
            scaler.step(optimizer)
            scaler.update()
            
            # BroadFace Memory Queue update & drift compensation
            broadface_queue.update(embeddings.detach(), labels)
            broadface_queue.compensate_weight_drift(old_weight, curricular_loss_fn.weight)
            
            running_loss += loss.item()
            
        scheduler.step()
        avg_loss = running_loss / max(1, len(train_loader))
        t_param = curricular_loss_fn.t.item()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch}/{cfg.train.epochs}] - Curricular Loss: {avg_loss:.4f} | EMA t: {t_param:.4f} | LR: {current_lr:.6f}")
        
    # Save Phase 1 Model Checkpoint
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.train.checkpoint_dir, "phase1_iresnet100_curricular.pt")
    torch.save({
        'backbone': backbone.state_dict(),
        'curricular_loss': curricular_loss_fn.state_dict(),
        'broadface_queue': broadface_queue.state_dict()
    }, ckpt_path)
    print(f"\n[Phase 1 Complete] Model checkpoint successfully saved to: {ckpt_path}")
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
    scaler = GradScaler(enabled=cfg.model.fp16)
    
    anet.train()
    for epoch in range(1, min(cfg.train.epochs, 5) + 1):
        running_loss = 0.0
        for images, attr_targets in train_loader:
            images, attr_targets = images.to(device), attr_targets.to(device)
            
            optimizer.zero_grad()
            with autocast(enabled=cfg.model.fp16):
                attr_logits, occ_mask, _ = anet(images)
                loss = criterion_bce(attr_logits, attr_targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"ANet Epoch [{epoch}/{min(cfg.train.epochs, 5)}] - Attribute BCE Loss: {avg_loss:.4f}")
        
    ckpt_path = os.path.join(cfg.train.checkpoint_dir, "phase2_anet_attributes.pt")
    torch.save({'anet': anet.state_dict()}, ckpt_path)
    print(f"[Phase 2 Complete] ANet attribute parser checkpoint saved to: {ckpt_path}")
    return anet

def main():
    parser = argparse.ArgumentParser(description="OccuPose-BroadDictNet Production Multi-Stage Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="", help="Path to training face dataset directory (e.g. WebFace-OCC)")
    parser.add_argument("--celeba_dir", type=str, default="", help="Path to CelebA attribute dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs per phase")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--fp16", action="store_true", help="Enable AMP FP16 mixed precision training")
    args = parser.parse_args()
    
    cfg = SystemConfig()
    if args.data_dir:
        cfg.dataset.train_data_dir = args.data_dir
    if args.celeba_dir:
        cfg.dataset.celeba_dir = args.celeba_dir
    cfg.train.checkpoint_dir = args.checkpoint_dir
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.learning_rate = args.lr
    cfg.model.fp16 = args.fp16 or torch.cuda.is_available()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running OccuPose-BroadDictNet SOTA Training Pipeline on: {device}")
    
    # Phase 1: Train IResNet-100 Backbone + CurricularFace + BroadFace
    backbone, num_classes = train_phase1_backbone_curricular(cfg, device)
    
    # Phase 2: Train ANet Attribute & Spatial Occlusion Parser
    anet = train_phase2_anet_attributes(cfg, device)
    
    print("\n========================================================================")
    print("✓ OccuPose-BroadDictNet Production Multi-Stage Training Pipeline Complete!")
    print("========================================================================")

if __name__ == "__main__":
    main()
