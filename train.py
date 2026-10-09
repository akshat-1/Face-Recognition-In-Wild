import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler

from config import SystemConfig
from dataset import WildFaceDataset, UnlabeledFaceDataset, CelebAAttributeDataset
from losses.broadface_queue import BroadFaceCurricularLoss
from losses.wing_loss import WingLoss
from models.backbone import ResNet100Backbone, vit_face_base, FasterRCNNBackbone
from models.detector import FasterRCNNFaceDetector
from models.anet_attribute import ANetAttributeParser
from models.sam_segmentor import SAMFaceSegmentor
from models.pim_frontalizer import PIMFrontalizationGAN
from models.gcn_cluster import GCNLinkPredictor

def get_backbone(cfg: SystemConfig):
    if cfg.model.backbone_type == "vit_face_base":
        return vit_face_base(embedding_dim=cfg.model.embedding_dim)
    elif cfg.model.backbone_type == "fasterrcnn_backbone":
        return FasterRCNNBackbone(embedding_dim=cfg.model.embedding_dim)
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

def train_phase1_backbone_curricular(cfg: SystemConfig, device: torch.device, is_ddp: bool = False, rank: int = 0, local_rank: int = 0, world_size: int = 1, resume: bool = False, force_retrain: bool = False):
    """
    Phase 1 Training: Backbone (IResNet-100 / ViT-Face) + CurricularFace Adaptive Loss + BroadFace Queue.
    Configured for Commercial SOTA standard (100 Epochs with periodic checkpointing).
    """
    if rank == 0:
        print(f"\n========================================================================")
        print(f"--- Starting Phase 1: {cfg.model.backbone_type.upper()} & CurricularFace SOTA Training ({cfg.train.epochs} Epochs) ---")
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
    detector = FasterRCNNFaceDetector(weights_dir=cfg.train.checkpoint_dir).to(device)
    detector.eval()
    sam_segmentor = SAMFaceSegmentor().to(device)
    sam_segmentor.eval()
    frontalizer = PIMFrontalizationGAN().to(device)
    frontalizer.eval()
    
    broadface_loss_fn = BroadFaceCurricularLoss(
        in_features=cfg.model.embedding_dim,
        num_classes=num_classes,
        scale_factor=cfg.loss.scale,
        margin=cfg.loss.margin,
        alpha=cfg.loss.ema_alpha,
        queue_size=cfg.loss.queue_size,
        compensate=True
    ).to(device)
    
    # Check if completed Phase 1 pretrained checkpoint exists
    phase1_ckpt = os.path.join(cfg.train.checkpoint_dir, "phase1_epoch_100.pt")
    if not os.path.exists(phase1_ckpt):
        phase1_ckpt = os.path.join(cfg.train.checkpoint_dir, "phase1_backbone_curricular.pt")
        
    if not force_retrain and os.path.exists(phase1_ckpt):
        try:
            ckpt_data = torch.load(phase1_ckpt, map_location=device)
            backbone.load_state_dict(ckpt_data['backbone'])
            saved_epoch = ckpt_data.get('epoch', 100)
            if rank == 0:
                print(f"\n[Phase 1 Pretrained] Loaded {saved_epoch}-epoch trained backbone weights from {phase1_ckpt}. Skipping Phase 1 training!\n", flush=True)
            return backbone, num_classes
        except Exception as e:
            if rank == 0:
                print(f"Warning: Could not load {phase1_ckpt}: {e}")

    # Resume from latest checkpoint if explicitly requested
    start_epoch = 1
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    latest_ckpt = os.path.join(cfg.train.checkpoint_dir, "latest_backbone.pt")
    if resume and os.path.exists(latest_ckpt):
        try:
            ckpt_data = torch.load(latest_ckpt, map_location=device)
            backbone.load_state_dict(ckpt_data['backbone'])
            if 'broadface_loss_fn' in ckpt_data:
                broadface_loss_fn.load_state_dict(ckpt_data['broadface_loss_fn'])
            start_epoch = ckpt_data.get('epoch', 1) + 1
            if rank == 0:
                print(f"✓ Resumed training state from checkpoint: {latest_ckpt} (Starting at Epoch {start_epoch})")
        except Exception as e:
            if rank == 0:
                print(f"Warning: Failed to load checkpoint {latest_ckpt}: {e}. Starting fresh.")

    if is_ddp and device.type == 'cuda':
        backbone = nn.parallel.DistributedDataParallel(backbone, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    optimizer = optim.SGD(
        list(backbone.parameters()) + list(broadface_loss_fn.parameters()),
        lr=cfg.train.learning_rate,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay
    )
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', cfg.train.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, last_epoch=start_epoch - 2 if start_epoch > 1 else -1)
    scaler = GradScaler('cuda', enabled=cfg.model.fp16 and device.type == 'cuda')
    
    backbone.train()
    broadface_loss_fn.train()
    
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
            
        if hasattr(train_dataset, "rescan"):
            new_count = train_dataset.rescan()
            if is_ddp:
                sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=cfg.train.batch_size,
                    sampler=sampler,
                    num_workers=cfg.dataset.num_workers,
                    pin_memory=True
                )
            
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            images, labels, _ = batch
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast('cuda', enabled=cfg.model.fp16 and device.type == 'cuda'):
                with torch.no_grad():
                    mask_sam = sam_segmentor(images)
                segmented_images = images * mask_sam
                
                # Check head pose yaw angle heuristic during Phase 1 training
                crop_aspect_ratio = images.shape[3] / float(images.shape[2] + 1e-5)
                estimated_yaw = (crop_aspect_ratio - 1.0) * 45.0
                if abs(estimated_yaw) > 20.0:
                    with torch.no_grad():
                        segmented_images = frontalizer(segmented_images, yaw_angle=estimated_yaw)
                        
                embeddings = backbone(segmented_images)
                loss = broadface_loss_fn(embeddings, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            
        scheduler.step()
        avg_loss = running_loss / max(1, len(train_loader))
        t_param = broadface_loss_fn.t.item()
        if rank == 0:
            print(f"Epoch [{epoch}/{cfg.train.epochs}] - BroadFace Curricular Loss: {avg_loss:.4f} | EMA t: {t_param:.4f}")
            
            # Periodic checkpointing every 10 epochs
            if epoch % 10 == 0 or epoch == cfg.train.epochs:
                backbone_state = backbone.module.state_dict() if hasattr(backbone, 'module') else backbone.state_dict()
                save_dict = {
                    'epoch': epoch,
                    'backbone': backbone_state,
                    'broadface_loss_fn': broadface_loss_fn.state_dict(),
                }
                periodic_path = os.path.join(cfg.train.checkpoint_dir, f"phase1_epoch_{epoch}.pt")
                torch.save(save_dict, periodic_path)
                torch.save(save_dict, latest_ckpt)
                print(f"✓ Saved periodic checkpoint to: {periodic_path}")
        
    if rank == 0:
        ckpt_path = os.path.join(cfg.train.checkpoint_dir, "phase1_backbone_curricular.pt")
        backbone_state = backbone.module.state_dict() if hasattr(backbone, 'module') else backbone.state_dict()
        torch.save({
            'epoch': cfg.train.epochs,
            'backbone': backbone_state,
            'broadface_loss_fn': broadface_loss_fn.state_dict(),
        }, ckpt_path)
        print(f"\n[Phase 1 Complete] Final Phase 1 Checkpoint saved to: {ckpt_path}")
    return backbone, num_classes

def train_phase2_anet_attributes(cfg: SystemConfig, backbone: nn.Module = None, device: torch.device = None, is_ddp: bool = False, rank: int = 0, local_rank: int = 0, world_size: int = 1, force_retrain: bool = False):
    """
    Phase 2 Training: ANet Multi-Task 40-Attribute Parser & Spatial Occlusion Attention.
    Trained across all datasets (unlabeled, labeled, celeba) with self-supervised spatial mask supervision.
    """
    if rank == 0:
        print(f"\n========================================================================")
        print(f"--- Starting Phase 2: ANet Multi-Task Attribute Parser Training ---")
        print(f"========================================================================\n")
    
    celeba_dataset = CelebAAttributeDataset(root_dir=cfg.dataset.celeba_dir, is_train=True)
    if is_ddp:
        sampler = DistributedSampler(celeba_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(celeba_dataset, batch_size=cfg.train.batch_size, sampler=sampler, num_workers=cfg.dataset.num_workers, pin_memory=True)
    else:
        sampler = None
        train_loader = DataLoader(celeba_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers, pin_memory=True)
    
    if rank == 0:
        print(f"Phase 2 Unified Attribute Dataset initialized: {len(celeba_dataset)} images across all datasets.")
    
    anet = ANetAttributeParser(num_attributes=40).to(device)
    sam_segmentor = SAMFaceSegmentor().to(device)
    frontalizer = PIMFrontalizationGAN().to(device)
    frontalizer.eval()
    
    # Check if completed Phase 2 pretrained checkpoint exists (skip only if not force_retrain)
    phase2_ckpt = os.path.join(cfg.train.checkpoint_dir, "phase2_anet_attributes.pt")
    if os.path.exists(phase2_ckpt) and not force_retrain:
        try:
            ckpt_data = torch.load(phase2_ckpt, map_location=device)
            anet.load_state_dict(ckpt_data['anet'])
            if rank == 0:
                print(f"\n[Phase 2 Pretrained] Loaded ANet attribute parser weights from {phase2_ckpt}. Skipping Phase 2 training!\n")
            return anet
        except Exception as e:
            if rank == 0:
                print(f"Warning: Could not load {phase2_ckpt}: {e}")
                
    if rank == 0:
        print(f"[Phase 2 Setup] Transferring Phase 1 identity backbone weights into ANet feature extractor...")
    if backbone is not None:
        anet.init_from_backbone(backbone)

    if is_ddp and device.type == 'cuda':
        anet = nn.parallel.DistributedDataParallel(anet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mask = nn.BCELoss()
    criterion_wing = WingLoss(w=10.0, epsilon=2.0).to(device)
    optimizer = optim.Adam(list(anet.parameters()) + list(sam_segmentor.parameters()), lr=1e-3, weight_decay=1e-4)
    
    phase2_epochs = min(cfg.train.epochs, 30)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-5)
    scaler = GradScaler('cuda', enabled=cfg.model.fp16 and device.type == 'cuda')
    
    anet.train()
    for epoch in range(1, phase2_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        running_loss = 0.0
        for step, (images, attr_targets, gt_masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            attr_targets = attr_targets.to(device, non_blocking=True)
            gt_masks = gt_masks.to(device, non_blocking=True)
            
            # Check head pose yaw angle heuristic in Phase 2
            crop_aspect_ratio = images.shape[3] / float(images.shape[2] + 1e-5)
            estimated_yaw = (crop_aspect_ratio - 1.0) * 45.0
            if abs(estimated_yaw) > 20.0:
                with torch.no_grad():
                    images = frontalizer(images, yaw_angle=estimated_yaw)
            
            optimizer.zero_grad()
            with autocast('cuda', enabled=cfg.model.fp16 and device.type == 'cuda'):
                attr_logits, occ_mask, _ = anet(images)
                sam_mask = sam_segmentor(images)
                loss_attr = criterion_bce(attr_logits, attr_targets)
                
            # BCELoss & WingLoss evaluated in float32 for numerical stability
            loss_mask = criterion_mask(occ_mask.float(), gt_masks.float())
            loss_sam = criterion_mask(sam_mask.float(), gt_masks.float())
            loss_wing = criterion_wing(occ_mask.float(), gt_masks.float())
            
            # Multi-Task Joint Loss: Attributes (BCE) + Spatial Mask (BCE) + SAM Mask (BCE) + Landmark Alignment (WingLoss)
            loss = loss_attr + 0.5 * loss_mask + 0.3 * loss_sam + 0.2 * loss_wing
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            
            if rank == 0 and (step % 50 == 0 or step == len(train_loader) - 1):
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else 1e-3
                print(f"ANet Epoch [{epoch}/{phase2_epochs}] - Step [{step}/{len(train_loader)}] - Loss: {loss.item():.4f} (Attr: {loss_attr.item():.4f}, Mask: {loss_mask.item():.4f}) | LR: {current_lr:.6f}", flush=True)
            
        scheduler.step()
        if rank == 0:
            print(f"✓ ANet Epoch [{epoch}/{phase2_epochs}] Complete - Avg Loss: {running_loss / max(1, len(train_loader)):.4f}\n", flush=True)
            if epoch % 10 == 0 or epoch == phase2_epochs:
                anet_state = anet.module.state_dict() if hasattr(anet, 'module') else anet.state_dict()
                periodic_path = os.path.join(cfg.train.checkpoint_dir, f"phase2_epoch_{epoch}.pt")
                torch.save({'epoch': epoch, 'anet': anet_state}, periodic_path)
                print(f"✓ Saved Phase 2 periodic checkpoint to: {periodic_path}\n", flush=True)
        
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
        gcn_predictor = nn.parallel.DistributedDataParallel(gcn_predictor, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
    optimizer_gcn = optim.Adam(gcn_predictor.parameters(), lr=1e-3)
    
    if rank == 0:
        print(f"Unlabeled wild face dataset loaded: {len(unlabeled_dataset)} images.")
    
    unwrapped_backbone = backbone.module if hasattr(backbone, 'module') else backbone
    unwrapped_backbone.eval()
    
    phase3_epochs = min(cfg.train.epochs, 10)
    for epoch in range(1, phase3_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
            
        if hasattr(unlabeled_dataset, "rescan"):
            unlabeled_dataset.rescan()
            if is_ddp:
                sampler = DistributedSampler(unlabeled_dataset, num_replicas=world_size, rank=rank, shuffle=True)
                unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg.train.batch_size, sampler=sampler, num_workers=cfg.dataset.num_workers)
            
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
            print(f"GCN Semi-Supervised Epoch [{epoch}/{phase3_epochs}] - High-Confidence Pseudo-Labeled Faces: {total_clustered}/{len(unlabeled_dataset)}")
        
    if rank == 0:
        ckpt_path = os.path.join(cfg.train.checkpoint_dir, "phase3_semi_gcn_predictor.pt")
        gcn_state = gcn_predictor.module.state_dict() if hasattr(gcn_predictor, 'module') else gcn_predictor.state_dict()
        torch.save({'gcn_predictor': gcn_state}, ckpt_path)
        print(f"[Phase 3 Complete] GCN Link Predictor checkpoint saved to: {ckpt_path}")

def main():
    parser = argparse.ArgumentParser(description="OccuPose-BroadDictNet Commercial SOTA Multi-Stage Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="", help="Path to labeled training dataset (ROF, LFW, WIDER)")
    parser.add_argument("--unlabeled_dir", type=str, default="", help="Path to unlabeled face dataset (FMD, COVID faces)")
    parser.add_argument("--celeba_dir", type=str, default="", help="Path to CelebA attribute dataset")
    parser.add_argument("--backbone", type=str, default="iresnet100", choices=["iresnet100", "vit_face_base"], help="Backbone type")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for Phase 1 backbone training (Commercial SOTA standard = 100)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--fp16", action="store_true", help="Enable AMP FP16")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if available")
    parser.add_argument("--retrain_phase1", action="store_true", help="Force retraining Phase 1 backbone from Epoch 1 (ignore old Phase 1 checkpoint)")
    parser.add_argument("--retrain_phase2", action="store_true", help="Force retraining Phase 2 ANet attribute parser (ignore old Phase 2 checkpoint)")
    parser.add_argument("--retrain_all", action="store_true", help="Force retraining ALL phases (Phase 1, Phase 2, Phase 3) from scratch")
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
        print(f"Running OccuPose-BroadDictNet Training Engine on: {device} (Backbone: {cfg.model.backbone_type}) | DDP: {is_ddp} (World Size: {world_size}) | Target Epochs: {cfg.train.epochs}")
    
    force_p1 = args.retrain_phase1 or args.retrain_all
    force_p2 = args.retrain_phase2 or args.retrain_all
    
    # Phase 1: Train Backbone + CurricularFace + BroadFace
    backbone, num_classes = train_phase1_backbone_curricular(cfg, device, is_ddp=is_ddp, rank=rank, local_rank=local_rank, world_size=world_size, resume=args.resume, force_retrain=force_p1)
    
    # Phase 2: Train ANet Attribute Parser
    anet = train_phase2_anet_attributes(cfg, backbone, device, is_ddp=is_ddp, rank=rank, local_rank=local_rank, world_size=world_size, force_retrain=force_p2)
    
    # Phase 3: Train GCN Semi-Supervised Link Predictor & Pseudo-Labeler on Unlabeled Faces
    train_phase3_semi_supervised_gcn(cfg, backbone, num_classes, device, is_ddp=is_ddp, rank=rank, local_rank=local_rank, world_size=world_size)
    
    if rank == 0:
        print("\n========================================================================")
        print("✓ OccuPose-BroadDictNet Commercial SOTA Multi-Stage Training Pipeline Complete!")
        print("========================================================================")
        
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
