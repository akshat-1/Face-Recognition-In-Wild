"""Training entry point for the corrected AQUA face pipeline.

Phase 0 trains the face detector from real ``bb_label`` YOLO boxes. Phase 1
trains identity embeddings from ``name_label`` face crops; detection labels do
not contain identity supervision and are therefore not mixed into that loss.
"""

import argparse
import os
import random
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

from config import SystemConfig
from dataset import BoundingBoxFaceDataset, CelebAAttributeDataset, UnlabeledFaceDataset, WildFaceDataset, get_default_transform
from losses.broadface_queue import BroadFaceCurricularLoss
from models.anet_attribute import ANetAttributeParser
from models.backbone import FasterRCNNBackbone, ResNet100Backbone, vit_face_base
from models.detector import TorchvisionFaceDetector
from models.gcn_cluster import GCNLinkPredictor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_backbone(cfg: SystemConfig):
    if cfg.model.backbone_type == "vit_face_base":
        return vit_face_base(embedding_dim=cfg.model.embedding_dim)
    if cfg.model.backbone_type == "fasterrcnn_backbone":
        return FasterRCNNBackbone(embedding_dim=cfg.model.embedding_dim)
    if cfg.model.backbone_type == "iresnet100":
        return ResNet100Backbone(embedding_dim=cfg.model.embedding_dim)
    raise ValueError(f"Unsupported backbone type: {cfg.model.backbone_type}")


def setup_ddp():
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ and torch.cuda.is_available()
    if is_ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, local_rank, world_size, is_ddp = 0, 0, 1, False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return is_ddp, rank, local_rank, world_size, device


class RecognitionModel(nn.Module):
    """Wrap backbone and classifier so DDP synchronizes both parameter sets."""

    def __init__(self, backbone: nn.Module, loss_fn: BroadFaceCurricularLoss):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        embeddings = self.backbone(images)
        return self.loss_fn(embeddings, labels), embeddings


def _state(module):
    raw = module.module if hasattr(module, "module") else module
    return raw.state_dict()


def _backbone_state(module):
    raw = module.module if hasattr(module, "module") else module
    if hasattr(raw, "backbone"):
        raw = raw.backbone
    return raw.state_dict()


def _save_phase1(path, epoch, backbone, loss_fn, optimizer, scheduler, scaler, cfg, class_to_idx):
    torch.save({
        "schema_version": 2,
        "kind": "recognition_backbone",
        "epoch": epoch,
        "backbone": _backbone_state(backbone),
        "backbone_state": _backbone_state(backbone),
        "broadface_loss_fn": loss_fn.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "model_config": asdict(cfg.model),
        "input_size": list(cfg.dataset.image_size),
        "loss_config": asdict(cfg.loss),
        "class_to_idx": dict(class_to_idx),
    }, path)


def _detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def train_phase0_face_detector(
    cfg: SystemConfig,
    device: torch.device,
    is_ddp: bool = False,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
    force_retrain: bool = False,
):
    """Train the class-agnostic face detector from AQUA ``bb_label`` targets."""
    if not cfg.dataset.bb_data_dir:
        if rank == 0:
            print("Phase 0 skipped: no bb_data_dir configured.")
        return None

    has_train_split = any(
        os.path.basename(os.path.normpath(current)).lower() == "train"
        for current, _, _ in os.walk(cfg.dataset.bb_data_dir)
    )
    dataset = BoundingBoxFaceDataset(
        root_dir=cfg.dataset.bb_data_dir,
        is_train=True,
        image_size=cfg.dataset.image_size,
        detection_mode=True,
        split="train" if has_train_split else None,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.detector_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.dataset.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_detection_collate,
    )
    checkpoint_dir = cfg.train.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "face_detector.pt")
    detector = TorchvisionFaceDetector(
        num_classes=2,
        pretrained_backbone=cfg.train.detector_pretrained_backbone,
    ).to(device)

    if os.path.isfile(checkpoint_path) and not force_retrain:
        detector.load_checkpoint(checkpoint_path)
        if rank == 0:
            print(f"Loaded completed bb_label detector checkpoint: {checkpoint_path}")
        return checkpoint_path

    if is_ddp:
        detector.model = nn.parallel.DistributedDataParallel(
            detector.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    optimizer = optim.SGD(
        detector.model.parameters(),
        lr=cfg.train.detector_learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scaler = GradScaler("cuda", enabled=cfg.model.fp16 and device.type == "cuda")
    for epoch in range(1, cfg.train.detector_epochs + 1):
        detector.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        running_loss = 0.0
        for images, targets in loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=cfg.model.fp16 and device.type == "cuda"):
                loss_dict = detector.detection_loss(images, targets)
                loss = sum(loss_value for loss_value in loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().item())
        if rank == 0:
            print(f"Phase 0 detector epoch {epoch}/{cfg.train.detector_epochs} loss={running_loss / max(1, len(loader)):.5f}")

    if rank == 0:
        raw_model = detector.model.module if hasattr(detector.model, "module") else detector.model
        torch.save({
            "schema_version": 2,
            "kind": "face_detector",
            "architecture": "fasterrcnn_resnet50_fpn",
            "num_classes": 2,
            "epoch": cfg.train.detector_epochs,
            "model_state": raw_model.state_dict(),
        }, checkpoint_path)
    if is_ddp and dist.is_initialized():
        dist.barrier()
    return checkpoint_path


def train_phase1_backbone_curricular(
    cfg: SystemConfig,
    device: torch.device,
    is_ddp: bool = False,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
    resume: bool = False,
    force_retrain: bool = False,
):
    """Train an identity backbone on real labelled face crops."""
    if tuple(cfg.dataset.image_size) != (112, 112):
        raise ValueError("Recognition backbones in this pipeline require image_size=(112,112)")
    dataset = WildFaceDataset(
        root_dir=cfg.dataset.train_data_dir,
        is_train=True,
        image_size=cfg.dataset.image_size,
        allow_synthetic=cfg.dataset.allow_synthetic_fixtures,
    )
    num_classes = len(dataset.class_to_idx)
    if num_classes <= 0:
        raise ValueError("The labelled dataset contains no identities")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.dataset.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=is_ddp,
    )

    backbone = get_backbone(cfg).to(device)
    loss_fn = BroadFaceCurricularLoss(
        in_features=cfg.model.embedding_dim,
        num_classes=num_classes,
        scale_factor=cfg.loss.scale,
        margin=cfg.loss.margin,
        alpha=cfg.loss.ema_alpha,
        queue_size=cfg.loss.queue_size,
        compensate=True,
    ).to(device)
    model = RecognitionModel(backbone, loss_fn).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.train.learning_rate,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.train.epochs))
    scaler = GradScaler("cuda", enabled=cfg.model.fp16 and device.type == "cuda")
    start_epoch = 1
    checkpoint_dir = cfg.train.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_path = os.path.join(checkpoint_dir, "recognition_backbone.pt")
    resume_path = os.path.join(checkpoint_dir, "latest_backbone.pt")

    if os.path.isfile(final_path) and not force_retrain and not resume:
        checkpoint = torch.load(final_path, map_location=device)
        state = checkpoint.get("backbone_state", checkpoint.get("backbone"))
        if state is None:
            raise ValueError(f"Invalid recognition checkpoint: {final_path}")
        backbone.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
        if rank == 0:
            print(f"Loaded completed Phase 1 checkpoint: {final_path}")
        return backbone, num_classes

    if resume and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        backbone.load_state_dict(checkpoint["backbone_state"])
        loss_fn.load_state_dict(checkpoint["broadface_loss_fn"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        running_loss = 0.0
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=cfg.model.fp16 and device.type == "cuda"):
                loss, embeddings = model(images, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.loss_fn.enqueue(embeddings.detach(), labels)
            running_loss += float(loss.detach().item())
        scheduler.step()
        if rank == 0:
            average = running_loss / max(1, len(loader))
            print(f"Phase 1 epoch {epoch}/{cfg.train.epochs} loss={average:.5f} t={loss_fn.t.item():.4f}")
            if epoch % 10 == 0 or epoch == cfg.train.epochs:
                _save_phase1(os.path.join(checkpoint_dir, f"phase1_epoch_{epoch}.pt"), epoch, model, loss_fn, optimizer, scheduler, scaler, cfg, dataset.class_to_idx)
                _save_phase1(resume_path, epoch, model, loss_fn, optimizer, scheduler, scaler, cfg, dataset.class_to_idx)

    if rank == 0:
        _save_phase1(final_path, cfg.train.epochs, model, loss_fn, optimizer, scheduler, scaler, cfg, dataset.class_to_idx)
    return model.module.backbone if hasattr(model, "module") else model.backbone, num_classes


def train_phase2_anet_attributes(cfg: SystemConfig, backbone=None, device=None, **kwargs):
    """Optional attribute training using only genuinely annotated CelebA rows."""
    if tuple(cfg.dataset.image_size) != (112, 112):
        raise ValueError("Attribute parser training requires image_size=(112,112)")
    rank = kwargs.get("rank", 0)
    dataset = CelebAAttributeDataset(root_dir=cfg.dataset.celeba_dir, image_size=cfg.dataset.image_size, is_train=True)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers)
    anet = ANetAttributeParser(num_attributes=40).to(device)
    if backbone is not None:
        anet.init_from_backbone(backbone)
    optimizer = optim.Adam(anet.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    epochs = min(cfg.train.epochs, 10)
    for epoch in range(1, epochs + 1):
        anet.train()
        running = 0.0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = anet(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().item())
        if rank == 0:
            print(f"Phase 2 epoch {epoch}/{epochs} loss={running / max(1, len(loader)):.5f}")
    if rank == 0:
        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
        torch.save({"schema_version": 2, "kind": "attribute_parser", "epoch": epochs, "anet": anet.state_dict()}, os.path.join(cfg.train.checkpoint_dir, "phase2_anet_attributes.pt"))
    return anet


class _CombinedIdentityDataset(Dataset):
    """Path-backed labelled/pseudo-labelled stream for Phase 3."""

    def __init__(self, samples, image_size):
        self.samples = list(samples)
        self.transform = get_default_transform(image_size, is_train=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.samples):
            raise IndexError(index)
        path, label, is_pseudo = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Could not decode Phase 3 image {path}: {exc}") from exc
        return self.transform(image), int(label), bool(is_pseudo)


def train_phase3_semi_supervised_gcn(cfg: SystemConfig, backbone, num_labeled_classes: int, device: torch.device, **kwargs):
    """Optional offline GCN adaptation over a globally indexed unlabeled stream."""
    rank = kwargs.get("rank", 0)
    if not cfg.dataset.unlabeled_data_dir:
        if rank == 0:
            print("Phase 3 skipped: no unlabeled_data_dir configured.")
        return
    dataset = UnlabeledFaceDataset(root_dir=cfg.dataset.unlabeled_data_dir, is_train=False, image_size=cfg.dataset.image_size)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers)
    raw_backbone = backbone.module if hasattr(backbone, "module") else backbone
    raw_backbone.eval()
    embeddings = []
    with torch.no_grad():
        for images, _, _ in loader:
            embeddings.append(raw_backbone(images.to(device)).cpu())
    all_embeddings = torch.cat(embeddings, dim=0)
    predictor = GCNLinkPredictor(feature_dim=cfg.model.embedding_dim, hidden_dim=256, k_neighbors=5).to(device)
    # Train the edge predictor on labelled identities before using it for
    # clustering. The global path falls back to a blockwise graph for very
    # large collections rather than constructing an N^2 hidden tensor.
    labelled = WildFaceDataset(root_dir=cfg.dataset.train_data_dir, is_train=False, image_size=cfg.dataset.image_size)
    labelled_loader = DataLoader(
        labelled,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )
    predictor.train()
    predictor_optimizer = optim.Adam(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(min(3, cfg.train.epochs)):
        for images, labels, _ in labelled_loader:
            if images.shape[0] < 2:
                continue
            with torch.no_grad():
                labelled_embeddings = raw_backbone(images.to(device))
            predictor_optimizer.zero_grad(set_to_none=True)
            edge_loss, _ = predictor(labelled_embeddings, labels.to(device))
            edge_loss.backward()
            predictor_optimizer.step()
    predictor.eval()
    pseudo_labels, active = predictor.generate_global_pseudo_labels(
        all_embeddings.to(device),
        confidence_threshold=cfg.loss.pseudo_label_threshold,
        start_class_idx=num_labeled_classes,
    )

    active_indices = torch.where(active.cpu())[0].tolist()
    pseudo_samples = [
        (dataset.samples[index], int(pseudo_labels[index].item()), True)
        for index in active_indices
        if dataset.samples[index] != "dummy_unlabeled"
    ]
    labelled_samples = [(path, label, False) for path, label in labelled.samples]
    joint_dataset = _CombinedIdentityDataset(labelled_samples + pseudo_samples, cfg.dataset.image_size)
    if len(joint_dataset) and pseudo_samples:
        total_classes = max(num_labeled_classes, max(label for _, label, _ in pseudo_samples) + 1)
        joint_loss_fn = BroadFaceCurricularLoss(
            in_features=cfg.model.embedding_dim,
            num_classes=total_classes,
            scale_factor=cfg.loss.scale,
            margin=cfg.loss.margin,
            alpha=cfg.loss.ema_alpha,
            queue_size=cfg.loss.queue_size,
            compensate=True,
        ).to(device)
        joint_optimizer = optim.SGD(
            list(raw_backbone.parameters()) + list(joint_loss_fn.parameters()),
            lr=cfg.train.learning_rate * 0.1,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay,
        )
        joint_loader = DataLoader(joint_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers)
        raw_backbone.train()
        for epoch in range(min(3, cfg.train.epochs)):
            for images, labels, is_pseudo in joint_loader:
                images, labels = images.to(device), labels.to(device)
                joint_optimizer.zero_grad(set_to_none=True)
                embeddings = raw_backbone(images)
                labelled_mask = ~is_pseudo.bool()
                pseudo_mask = is_pseudo.bool()
                total_loss = embeddings.sum() * 0.0
                if labelled_mask.any():
                    total_loss = total_loss + joint_loss_fn(embeddings[labelled_mask], labels[labelled_mask])
                if pseudo_mask.any():
                    total_loss = total_loss + cfg.loss.lambda_semi * joint_loss_fn(embeddings[pseudo_mask], labels[pseudo_mask])
                total_loss.backward()
                joint_optimizer.step()
                to_enqueue = embeddings.detach()
                joint_loss_fn.enqueue(to_enqueue, labels)
        raw_backbone.eval()
    checkpoint_path = os.path.join(cfg.train.checkpoint_dir, "phase3_semi_gcn_predictor.pt")
    if rank == 0:
        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
        torch.save({
            "schema_version": 2,
            "kind": "gcn_adapter",
            "gcn_predictor": predictor.state_dict(),
            "pseudo_labels": pseudo_labels.cpu(),
            "active_mask": active.cpu(),
        }, checkpoint_path)
        torch.save({
            "schema_version": 2,
            "kind": "recognition_backbone",
            "backbone_state": raw_backbone.state_dict(),
            "broadface_loss_fn": joint_loss_fn.state_dict() if pseudo_samples and len(joint_dataset) else None,
            "epoch": min(3, cfg.train.epochs),
        }, os.path.join(cfg.train.checkpoint_dir, "phase3_adapted_backbone.pt"))
        print(f"Phase 3 generated {int(active.sum())} globally indexed pseudo-labelled samples; saved {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="OccuPose-BroadDictNet training")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--bb_dir", default=None, help="Path to AQUA bb_label YOLO/Darknet detection dataset")
    parser.add_argument("--unlabeled_dir", default=None)
    parser.add_argument("--celeba_dir", default=None)
    parser.add_argument("--backbone", choices=["iresnet100", "vit_face_base", "fasterrcnn_backbone"], default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--detector_batch_size", type=int, default=None)
    parser.add_argument("--detector_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retrain_phase1", action="store_true")
    parser.add_argument("--retrain_phase2", action="store_true")
    parser.add_argument("--retrain_all", action="store_true")
    parser.add_argument("--allow_synthetic_fixtures", action="store_true")
    args = parser.parse_args()

    is_ddp, rank, local_rank, world_size, device = setup_ddp()
    cfg = SystemConfig()
    if args.data_dir is not None: cfg.dataset.train_data_dir = args.data_dir
    if args.bb_dir is not None: cfg.dataset.bb_data_dir = args.bb_dir
    if args.unlabeled_dir is not None: cfg.dataset.unlabeled_data_dir = args.unlabeled_dir
    if args.celeba_dir is not None: cfg.dataset.celeba_dir = args.celeba_dir
    if args.backbone is not None: cfg.model.backbone_type = args.backbone
    if args.checkpoint_dir is not None: cfg.train.checkpoint_dir = args.checkpoint_dir
    if args.epochs is not None: cfg.train.epochs = args.epochs
    if args.batch_size is not None: cfg.train.batch_size = args.batch_size
    if args.detector_batch_size is not None: cfg.train.detector_batch_size = args.detector_batch_size
    if args.detector_epochs is not None: cfg.train.detector_epochs = args.detector_epochs
    if args.lr is not None: cfg.train.learning_rate = args.lr
    cfg.model.fp16 = args.fp16 or (device.type == "cuda")
    cfg.dataset.allow_synthetic_fixtures = args.allow_synthetic_fixtures
    seed_everything(cfg.train.seed + rank)

    train_phase0_face_detector(
        cfg,
        device,
        is_ddp=is_ddp,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        force_retrain=args.retrain_all,
    )

    backbone, num_classes = train_phase1_backbone_curricular(
        cfg, device, is_ddp=is_ddp, rank=rank, local_rank=local_rank,
        world_size=world_size, resume=args.resume, force_retrain=args.retrain_phase1 or args.retrain_all,
    )
    if cfg.dataset.celeba_dir:
        train_phase2_anet_attributes(cfg, backbone, device, rank=rank, force_retrain=args.retrain_phase2 or args.retrain_all)
    else:
        if rank == 0:
            print("Phase 2 skipped: no celeba_dir configured.")
    if cfg.dataset.unlabeled_data_dir:
        train_phase3_semi_supervised_gcn(cfg, backbone, num_classes, device, rank=rank)
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
