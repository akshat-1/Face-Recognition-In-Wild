import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import WildFaceDataset, CelebAAttributeDataset
from losses.curricular_loss import CurricularFaceLoss
from losses.broadface_queue import BroadFaceMemoryQueue
from models.backbone import ResNet100Backbone
from models.anet_attribute import ANetAttributeParser
from models.ddrc_solver import DDRCClassifier

def train_phase1_backbone_curricular(args, device):
    """
    Phase 1 Training: Feature Backbone + CurricularFace Adaptive Loss + BroadFace Memory Queue.
    Operates on WebFace-OCC or general face dataset.
    """
    print(f"\n========================================================")
    print(f"--- Starting Phase 1: Backbone & CurricularFace Training ---")
    print(f"========================================================\n")
    
    # Dataset and DataLoader
    train_dataset = WildFaceDataset(root_dir=args.data_dir, is_train=True, num_synthetic_samples=500)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    num_classes = len(train_dataset.class_to_idx) if train_dataset.class_to_idx else args.num_classes
    print(f"Training dataset loaded: {len(train_dataset)} samples across {num_classes} identities.")
    
    # Models
    backbone = ResNet100Backbone(embedding_dim=512, layers=(3, 4, 6, 3)).to(device)
    curricular_loss_fn = CurricularFaceLoss(in_features=512, num_classes=num_classes, s=64.0, m=0.5).to(device)
    broadface_queue = BroadFaceMemoryQueue(queue_size=2048, feature_dim=512).to(device)
    
    # Optimizer & LR Scheduler
    optimizer = optim.SGD(
        list(backbone.parameters()) + list(curricular_loss_fn.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    backbone.train()
    curricular_loss_fn.train()
    
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            embeddings = backbone(images)
            loss = curricular_loss_fn(embeddings, labels)
            
            # Backward & optimizer step
            optimizer.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                old_weight = curricular_loss_fn.weight.clone()
                
            optimizer.step()
            
            # BroadFace Queue update & drift compensation
            broadface_queue.update(embeddings.detach(), labels)
            broadface_queue.compensate_weight_drift(old_weight, curricular_loss_fn.weight)
            
            running_loss += loss.item()
            
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        t_param = curricular_loss_fn.t.item()
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {avg_loss:.4f} | Curricular t: {t_param:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
    # Save Phase 1 Checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "phase1_backbone_curricular.pt")
    torch.save({
        'backbone': backbone.state_dict(),
        'curricular_loss': curricular_loss_fn.state_dict(),
        'broadface_queue': broadface_queue.state_dict()
    }, ckpt_path)
    print(f"\n[Phase 1 Complete] Model checkpoint saved to: {ckpt_path}")
    return backbone, num_classes

def train_phase2_anet_attributes(args, device):
    """
    Phase 2 Training: ANet Multi-Task 40-Attribute Parser & Spatial Occlusion Attention.
    """
    print(f"\n========================================================")
    print(f"--- Starting Phase 2: ANet Multi-Task Attribute Parser ---")
    print(f"========================================================\n")
    
    celeba_dataset = CelebAAttributeDataset(is_train=True, num_synthetic_samples=500)
    train_loader = DataLoader(celeba_dataset, batch_size=args.batch_size, shuffle=True)
    
    anet = ANetAttributeParser(num_attributes=40).to(device)
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(anet.parameters(), lr=1e-3, weight_decay=1e-4)
    
    anet.train()
    for epoch in range(1, min(args.epochs, 3) + 1):
        running_loss = 0.0
        for images, attr_targets in train_loader:
            images, attr_targets = images.to(device), attr_targets.to(device)
            
            attr_logits, occ_mask, _ = anet(images)
            loss = criterion_bce(attr_logits, attr_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"ANet Epoch [{epoch}/{min(args.epochs, 3)}] - Attribute BCE Loss: {running_loss / len(train_loader):.4f}")
        
    ckpt_path = os.path.join(args.checkpoint_dir, "phase2_anet_attributes.pt")
    torch.save({'anet': anet.state_dict()}, ckpt_path)
    print(f"[Phase 2 Complete] ANet checkpoint saved to: {ckpt_path}")
    return anet

def main():
    parser = argparse.ArgumentParser(description="OccuPose-BroadDictNet Multi-Stage Training Pipeline")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for training face dataset (e.g. WebFace-OCC)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per phase")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--num_classes", type=int, default=50, help="Number of identity classes")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running OccuPose-BroadDictNet Training Pipeline on: {device}")
    
    # Phase 1: Train Backbone + CurricularFace + BroadFace
    backbone, num_classes = train_phase1_backbone_curricular(args, device)
    
    # Phase 2: Train ANet Attribute & Spatial Occlusion Parser
    anet = train_phase2_anet_attributes(args, device)
    
    print("\n========================================================")
    print("✓ OccuPose-BroadDictNet Multi-Stage Training Complete!")
    print("========================================================")

if __name__ == "__main__":
    main()
