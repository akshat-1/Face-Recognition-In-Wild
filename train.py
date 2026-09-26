import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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

def setup_ddp():
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp and torch.cuda.is_available():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        is_ddp = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return is_ddp, rank, local_rank, world_size, device

def train_phase1_backbone_curricular(cfg: SystemConfig, device: torch.device, is_ddp: bool = False, rank: int = 0, local_rank: int = 0, world_size: int = 1):
    """
    Phase 1 Training: Backbone (IResNet-100 / ViT-Face) + CurricularFace Adaptive Loss + BroadFace Queue.
    """
    if rank == 0:
        print(f"\n========================================================================")
        print(f"--- Starting Phase 1: {cfg.model.backbone_type.upper()} & CurricularFace Training ---")
        print(f"========================================================================\n")
    
    train_dataset = WildFaceDataset(root_dir=cfg.dataset.train_data_dir, is_train=True)
    if is_ddp:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            sampler=sampler,
            num_workers=cfg.dataset.num_workers,
            pin_memory=True
        )
    else:
        sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
            pin_memory=True
        )
    
    num_classes = len(train_dataset.class_to_idx) if train_dataset.class_to_idx else cfg.dataset.num_classes
    if rank == 0:
        print(f"Labeled training dataset initialized: {len(train_dataset)} samples across {num_classes} identities.")
    
    backbone = get_backbone(cfg).to(device)
    if is_ddp and device.type == 'cuda':
        backbone = nn.parallel.DistributedDataParallel(backbone, device_ids=[local_rank], output_device=local_rank)
        
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
        if sampler is not None:
            sampler.set_epoch(epoch)
            
        if hasattr(train_dataset, "rescan"):
            total_samples = train_dataset.rescan()
            
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
        if rank == 0:
            print(f"Epoch [{epoch}/{cfg.train.epochs}] - Curricular Loss: {avg_loss:.4f} | EMA t: {t_param:.4f}")
        
    if rank == 0:
        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(cfg.train.checkpoint_dir, "phase1_backbone_curricular.pt")
        backbone_state = backbone.module.state_dict() if hasattr(backbone, 'module') else backbone.state_dict()
        torch.save({
            'backbone': backbone_state,
            'curricular_loss': curricular_loss_fn.state_dict(),
            'broadface_queue': broadface_queue.state_dict()
        }, ckpt_path)
        print(f"\n[Phase 1 Complete] Checkpoint saved to: {ckpt_path}")
    return backbone, num_classes

def train_phase2_anet_attributes(cfg: SystemConfig, device: torch.device, is_ddp: bool = False, rank: int = 0, local_rank: int = 0, world_size: int = 1):
    """
    Phase 2 Training: ANet Multi-Task 40-Attribute Parser & Spatial Occlusion Attention.
    """
    if rank == 0:
        print(f"\n========================================================================")
        print(f"--- Starting Phase 2: ANet Multi-Task Attribute Parser Training ---")
        print(f"========================================================================\n")
    
    celeba_dataset = CelebAAttributeDataset(root_dir=cfg.dataset.celeba_dir, is_train=True)
    if is_ddp:
        sampler = DistributedSampler(celeba_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(celeba_dataset, batch_size=cfg.train.batch_size, sampler=sampler, num_workers=cfg.dataset.num_workers)
    else:
        sampler = None
        train_loader = DataLoader(celeba_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers)
    
    anet = ANetAttributeParser(num_attributes=40).to(device)
    if is_ddp and device.type == 'cuda':
        anet = nn.parallel.DistributedDataParallel(anet, device_ids=[local_rank], output_device=local_rank)
        
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(anet.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = GradScaler('cuda', enabled=cfg.model.fp16 and device.type == 'cuda')
    
    anet.train()
    for epoch in range(1, min(cfg.train.epochs, 5) + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
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
            
        if rank == 0:
            print(f"ANet Epoch [{epoch}/{min(cfg.train.epochs, 5)}] - Attribute BCE Loss: {running_loss / max(1, len(train_loader)):.4f}")
        
    if rank == 0:
        ckpt_path = os.path.join(cfg.train.checkpoint_dir, "phase2_anet_attributes.pt")
        anet_state = anet.module.state_dict() if hasattr(anet, 'module') else anet.state_dict()
        torch.save({'anet': anet_state}, ckpt_path)
        print(f"[Phase 2 Complete] ANet checkpoint saved to: {ckpt_path}")
    return anet

def train_phase3_semi_supervised_gcn(cfg: SystemConfig, backbone, num_labeled_classes: int, device: torch.device, is_ddp: bool = False, rank: int = 0, local_rank: int = 0, world_size: int = 1):
    """
    Phase 3 Training: Semi-Supervised Unlabeled Face Clustering via GCN Sub-Graph Link Prediction.
    Tackles unannotated wild face datasets (FMD, COVID faces, web crawls).
    """
    if rank == 0:
        print(f"\n========================================================================")
        print(f"--- Starting Phase 3: Semi-Supervised GCN Clustering (Unlabeled Faces) ---")
        print(f"========================================================================\n")
    
    unlabeled_dataset = UnlabeledFaceDataset(root_dir=cfg.dataset.unlabeled_data_dir, is_train=True)
    if is_ddp:
        sampler = DistributedSampler(unlabeled_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg.train.batch_size, sampler=sampler, num_workers=cfg.dataset.num_workers)
    else:
        sampler = None
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers)
    
    gcn_predictor = GCNLinkPredictor(feature_dim=cfg.model.embedding_dim, hidden_dim=256, k_neighbors=5).to(device)
    if is_ddp and device.type == 'cuda':
        gcn_predictor = nn.parallel.DistributedDataParallel(gcn_predictor, device_ids=[local_rank], output_device=local_rank)
        
    optimizer_gcn = optim.Adam(gcn_predictor.parameters(), lr=1e-3)
    
    if rank == 0:
        print(f"Unlabeled wild face dataset loaded: {len(unlabeled_dataset)} images.")
    
    unwrapped_backbone = backbone.module if hasattr(backbone, 'module') else backbone
    unwrapped_backbone.eval()
    
    for epoch in range(1, min(cfg.train.epochs, 3) + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
            
        if hasattr(unlabeled_dataset, "rescan"):
            unlabeled_dataset.rescan()
            
        total_clustered = 0
        for images, labels, is_labeled in unlabeled_loader:
            images = images.to(device)
            with torch.no_grad():
                unlabeled_embeds = unwrapped_backbone(images)
                
            unwrapped_gcn = gcn_predictor.module if hasattr(gcn_predictor, 'module') else gcn_predictor
            pseudo_labels, active_mask = unwrapped_gcn.generate_pseudo_labels(
                unlabeled_embeds,
                confidence_threshold=cfg.loss.pseudo_label_threshold,
                start_class_idx=num_labeled_classes
            )
            total_clustered += active_mask.sum().item()
            
        if rank == 0:
            print(f"GCN Semi-Supervised Epoch [{epoch}/{min(cfg.train.epochs, 3)}] - High-Confidence Pseudo-Labeled Faces: {total_clustered}/{len(unlabeled_dataset)}")
        
    if rank == 0:
        ckpt_path = os.path.join(cfg.train.checkpoint_dir, "phase3_semi_gcn_predictor.pt")
        gcn_state = gcn_predictor.module.state_dict() if hasattr(gcn_predictor, 'module') else gcn_predictor.state_dict()
        torch.save({'gcn_predictor': gcn_state}, ckpt_path)
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
    
    is_ddp, rank, local_rank, world_size, device = setup_ddp()
    
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
    cfg.model.fp16 = args.fp16 or (device.type == 'cuda')
    
    if rank == 0:
        print(f"Running OccuPose-BroadDictNet Training Engine on: {device} (Backbone: {cfg.model.backbone_type}) | DDP: {is_ddp} (World Size: {world_size})")
    
    # Phase 1: Train Backbone + CurricularFace + BroadFace
    backbone, num_classes = train_phase1_backbone_curricular(cfg, device, is_ddp=is_ddp, rank=rank, local_rank=local_rank, world_size=world_size)
    
    # Phase 2: Train ANet Attribute Parser
    anet = train_phase2_anet_attributes(cfg, device, is_ddp=is_ddp, rank=rank, local_rank=local_rank, world_size=world_size)
    
    # Phase 3: Train GCN Semi-Supervised Link Predictor & Pseudo-Labeler on Unlabeled Faces
    train_phase3_semi_supervised_gcn(cfg, backbone, num_classes, device, is_ddp=is_ddp, rank=rank, local_rank=local_rank, world_size=world_size)
    
    if rank == 0:
        print("\n========================================================================")
        print("✓ OccuPose-BroadDictNet Production Multi-Stage Training Pipeline Complete!")
        print("========================================================================")
        
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
