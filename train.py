"""Training entry point for the corrected AQUA face pipeline.

Phase 0 trains the face detector from real ``bb_label`` YOLO boxes. Phase 1
trains identity embeddings from ``name_label`` face crops plus the explicit
nested-WIDER folder-labelled crops. Generic WIDER folder labels are kept
separate from YOLO detection classes.
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
from dataset import (
    BoundingBoxFaceDataset,
    CelebAAttributeDataset,
    CombinedIdentityDataset,
    DetectorCropDataset,
    UnlabeledFaceDataset,
    WiderFolderLabeledFaceDataset,
    WildFaceDataset,
    find_wider_special_root,
    get_default_transform,
)
from losses.broadface_queue import BroadFaceCurricularLoss
from models.anet_attribute import ANetAttributeParser
from models.backbone import FasterRCNNBackbone, ResNet100Backbone, vit_face_base
from models.detector import TorchvisionFaceDetector
from models.ddrc_solver import DDRCClassifier
from models.gcn_cluster import GCNLinkPredictor


def live_print(*args, **kwargs):
    """Print immediately so PBS/tee log followers see training progress."""
    kwargs["flush"] = True
    print(*args, **kwargs)


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


def resolve_wider_identity_root(cfg: SystemConfig):
    """Return the explicitly configured or nested AQUA WIDER label source."""
    if cfg.dataset.wider_identity_data_dir:
        return cfg.dataset.wider_identity_data_dir
    if cfg.dataset.unlabeled_data_dir:
        return find_wider_special_root(cfg.dataset.unlabeled_data_dir)
    return None


def build_identity_dataset(cfg: SystemConfig, is_train: bool):
    """Build name_label recognition data plus optional WIDER folder crops."""
    identity_dataset = WildFaceDataset(
        root_dir=cfg.dataset.train_data_dir,
        is_train=is_train,
        image_size=cfg.dataset.image_size,
        allow_synthetic=cfg.dataset.allow_synthetic_fixtures,
    )
    wider_root = resolve_wider_identity_root(cfg)
    if not wider_root:
        return CombinedIdentityDataset(
            [identity_dataset],
            image_size=cfg.dataset.image_size,
            is_train=is_train,
        )
    wider_dataset = WiderFolderLabeledFaceDataset(
        root_dir=wider_root,
        is_train=is_train,
        image_size=cfg.dataset.image_size,
    )
    combined = CombinedIdentityDataset(
        [identity_dataset, wider_dataset],
        image_size=cfg.dataset.image_size,
        is_train=is_train,
    )
    live_print(
        "[Recognition data] name_label=%d samples, WIDER folder-labelled=%d samples, "
        "total=%d classes=%d" % (
            len(identity_dataset),
            len(wider_dataset),
            len(combined),
            len(combined.class_to_idx),
        )
    )
    return combined


def load_backbone_for_adaptation(cfg: SystemConfig, device: torch.device):
    """Load an existing recognition artifact without retraining Phase 1."""
    dataset = build_identity_dataset(cfg, is_train=False)
    # Prioritize clean Phase 1 recognition_backbone.pt over phase3_adapted_backbone.pt
    candidates = [
        os.path.join(cfg.train.checkpoint_dir, "recognition_backbone.pt"),
        os.path.join(cfg.train.checkpoint_dir, "phase3_adapted_backbone.pt"),
    ]
    checkpoint_path = next((path for path in candidates if os.path.isfile(path)), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No recognition checkpoint found in {cfg.train.checkpoint_dir}; "
            "run --phase recognition first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != 2 or checkpoint.get("kind") != "recognition_backbone":
        raise ValueError(f"Not a recognition checkpoint: {checkpoint_path}")
    saved_config = checkpoint.get("model_config", {})
    if saved_config.get("backbone_type"):
        cfg.model.backbone_type = saved_config["backbone_type"]
    if saved_config.get("embedding_dim"):
        cfg.model.embedding_dim = int(saved_config["embedding_dim"])
    saved_alignment = checkpoint.get("recognition_alignment", saved_config.get("recognition_alignment"))
    if saved_alignment:
        cfg.model.recognition_alignment = saved_alignment
    backbone = get_backbone(cfg).to(device)
    state = checkpoint.get("backbone_state", checkpoint.get("backbone"))
    if not state:
        raise ValueError(f"Recognition checkpoint contains no backbone state: {checkpoint_path}")
    backbone.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    backbone.eval()
    live_print(f"Loaded recognition backbone from {checkpoint_path}")
    return backbone, len(dataset.class_to_idx)


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
        "recognition_alignment": cfg.model.recognition_alignment,
        "loss_config": asdict(cfg.loss),
        "class_to_idx": dict(class_to_idx),
    }, path)


def _detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def enroll_final_gallery(cfg: SystemConfig, backbone: nn.Module, device: torch.device):
    """Enroll the final trained recognition gallery using ONLY name_label identities."""
    dataset = WildFaceDataset(
        root_dir=cfg.dataset.train_data_dir,
        is_train=False,
        image_size=cfg.dataset.image_size,
        allow_synthetic=cfg.dataset.allow_synthetic_fixtures,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )

    embeddings = []
    labels = []

    raw_backbone = backbone.module if hasattr(backbone, "module") else backbone
    raw_backbone.eval()

    with torch.no_grad():
        for images, batch_labels, _ in loader:
            embeddings.append(raw_backbone(images.to(device)).cpu())
            labels.append(batch_labels)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0).long()

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    unique_labels = torch.unique(labels, sorted=True)
    class_names = [idx_to_class[int(i.item())] for i in unique_labels]

    classifier = DDRCClassifier(
        feature_dim=cfg.model.embedding_dim,
        num_classes=len(class_names),
    ).to(device)

    classifier.enroll_gallery(embeddings, labels, class_names=class_names)

    save_path = os.path.join(cfg.train.checkpoint_dir, "final_gallery.pt")
    torch.save({
        "classifier_state": classifier.state_dict(),
        "class_to_idx": dataset.class_to_idx,
        "class_names": class_names,
        "num_classes": len(class_names),
        "atoms_per_class": classifier.atoms_per_class,
        "embeddings": embeddings,
        "labels": labels,
    }, save_path)

    stale_cache = os.path.join(cfg.train.checkpoint_dir, "gallery_embeddings.pt")
    if os.path.isfile(stale_cache):
        try:
            os.remove(stale_cache)
        except OSError:
            pass

    live_print(
        f"Final gallery enrolled ({len(class_names)} active identities out of "
        f"{len(dataset.class_to_idx)} folders, {embeddings.shape[0]} samples) -> {save_path}"
    )

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
    wider_root = cfg.dataset.wider_identity_data_dir or find_wider_special_root(cfg.dataset.unlabeled_data_dir)
    detector_root = cfg.dataset.bb_data_dir or wider_root
    if not detector_root:
        if rank == 0:
            live_print("Phase 0 skipped: no bb_data_dir or nested WIDER detector data configured.")
        return None

    if cfg.train.detector_split not in {"all", "train"}:
        raise ValueError("detector_split must be 'all' or 'train'")
    include_wider = os.environ.get("AQUA_DETECTOR_INCLUDE_WIDER", "1").lower() not in {"0", "false", "no"}
    dataset = BoundingBoxFaceDataset(
        root_dir=detector_root,
        is_train=True,
        image_size=cfg.dataset.image_size,
        detection_mode=True,
        split=None if cfg.train.detector_split == "all" else "train",
        include_wider=include_wider,
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
    if rank == 0:
        live_print(f"DEBUG: Phase 0 looking for detector at {checkpoint_path}")
    existing_checkpoint = os.path.isfile(checkpoint_path) and not force_retrain
    detector = TorchvisionFaceDetector(
        checkpoint_path=checkpoint_path if existing_checkpoint else None,
        num_classes=2,
        pretrained_backbone=cfg.train.detector_pretrained_backbone and not existing_checkpoint,
        detector_min_size=cfg.train.detector_min_size,
        detector_max_size=cfg.train.detector_max_size,
    ).to(device)

    if existing_checkpoint:
        if rank == 0:
            live_print(f"Loaded completed detector checkpoint: {checkpoint_path}")
        return checkpoint_path

    if rank == 0 and wider_root:
        live_print(f"Phase 0 includes nested WIDER YOLO boxes: {wider_root}")
    if rank == 0:
        live_print(
            f"Phase 0 detector backend={detector.backend} architecture={detector.architecture} "
            f"image_min={cfg.train.detector_min_size} image_max={cfg.train.detector_max_size} "
            f"include_wider={include_wider}"
        )

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
        for step, (images, targets) in enumerate(loader, start=1):
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
            if rank == 0 and (step == 1 or step % max(1, cfg.train.log_interval) == 0):
                live_print(
                    f"Phase 0 detector epoch {epoch}/{cfg.train.detector_epochs} "
                    f"step {step}/{len(loader)} loss={float(loss.detach().item()):.5f}"
                )
        if rank == 0:
            live_print(f"Phase 0 detector epoch {epoch}/{cfg.train.detector_epochs} loss={running_loss / max(1, len(loader)):.5f}")
            torch.save({
                "schema_version": 2,
                "kind": "face_detector_progress",
                "architecture": detector.architecture,
                "backend": detector.backend,
                "num_classes": 2,
                "epoch": epoch,
                "model_state": (detector.model.module if hasattr(detector.model, "module") else detector.model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }, os.path.join(checkpoint_dir, "phase0_detector_latest.pt"))

    if rank == 0:
        raw_model = detector.model.module if hasattr(detector.model, "module") else detector.model
        torch.save({
            "schema_version": 2,
            "kind": "face_detector",
            "architecture": detector.architecture,
            "backend": detector.backend,
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
    dataset = build_identity_dataset(cfg, is_train=True)
    num_classes = len(dataset.class_to_idx)
    if num_classes <= 0:
        raise ValueError("The labelled dataset contains no identities")
    if cfg.train.batch_size < 2:
        raise ValueError("Recognition batch_size must be at least 2 because the embedding heads use BatchNorm")
    if len(dataset) < cfg.train.batch_size:
        raise ValueError(
            f"Recognition dataset has {len(dataset)} samples, fewer than batch_size={cfg.train.batch_size}; "
            "use a smaller batch_size or provide more identity images."
        )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.dataset.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
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
    if rank == 0:
        live_print(f"DEBUG: Phase 1 looking for final backbone at {final_path}, latest at {resume_path}")

    if os.path.isfile(final_path) and not force_retrain and not resume:
        checkpoint = torch.load(final_path, map_location=device, weights_only=False)
        state = checkpoint.get("backbone_state", checkpoint.get("backbone"))
        if state is None:
            raise ValueError(f"Invalid recognition checkpoint: {final_path}")
        backbone.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
        if rank == 0:
            live_print(f"Loaded completed Phase 1 checkpoint: {final_path}")
        return backbone, num_classes

    if resume and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
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
        for step, (images, labels, _) in enumerate(loader, start=1):
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
            if rank == 0 and (step == 1 or step % max(1, cfg.train.log_interval) == 0):
                live_print(
                    f"Phase 1 epoch {epoch}/{cfg.train.epochs} "
                    f"step {step}/{len(loader)} loss={float(loss.detach().item()):.5f}"
                )
        scheduler.step()
        if rank == 0:
            average = running_loss / max(1, len(loader))
            live_print(f"Phase 1 epoch {epoch}/{cfg.train.epochs} loss={average:.5f} t={loss_fn.t.item():.4f}")
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
    force_retrain = bool(kwargs.get("force_retrain", False))
    checkpoint_path = os.path.join(cfg.train.checkpoint_dir, "phase2_anet_attributes.pt")
    if os.path.isfile(checkpoint_path) and not force_retrain:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("schema_version") != 2 or checkpoint.get("kind") != "attribute_parser":
            raise ValueError(f"Unsupported Phase 2 checkpoint: {checkpoint_path}")
        anet = ANetAttributeParser(num_attributes=40).to(device)
        state = checkpoint.get("anet")
        if not state:
            raise ValueError(f"Phase 2 checkpoint has no attribute parser state: {checkpoint_path}")
        anet.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
        anet.eval()
        if rank == 0:
            live_print(f"Loaded completed Phase 2 checkpoint: {checkpoint_path}")
        return anet

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
            live_print(f"Phase 2 epoch {epoch}/{epochs} loss={running / max(1, len(loader)):.5f}")
            torch.save({
                "schema_version": 2,
                "kind": "attribute_parser_progress",
                "epoch": epoch,
                "epochs": epochs,
                "anet": anet.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(cfg.train.checkpoint_dir, "phase2_anet_attributes_latest.pt"))
    if rank == 0:
        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
        torch.save({"schema_version": 2, "kind": "attribute_parser", "epoch": epochs, "anet": anet.state_dict()}, os.path.join(cfg.train.checkpoint_dir, "phase2_anet_attributes.pt"))
    return anet


class _CombinedIdentityDataset(Dataset):
    """Path-backed labelled/pseudo-labelled stream for Phase 3."""

    def __init__(self, samples, image_size, validation_cache_path=None):
        self.transform = get_default_transform(image_size, is_train=True)
        self.samples = []
        skipped = 0
        if validation_cache_path and os.path.isfile(validation_cache_path):
            try:
                cached = torch.load(validation_cache_path, map_location="cpu", weights_only=False)
                if cached.get("schema_version") == 1:
                    cached_valid = set(cached.get("valid_paths", []))
                    cached_invalid = set(cached.get("invalid_paths", []))
                    for sample in samples:
                        path = sample[0]
                        if path in cached_valid:
                            self.samples.append(sample)
                        elif path in cached_invalid:
                            skipped += 1
                    if len(self.samples) + skipped == len(samples):
                        if skipped:
                            live_print(f"[Phase 3] Loaded validation cache; skipped {skipped} corrupt or unreadable labelled/pseudo samples")
                        return
            except Exception as exc:
                live_print(f"[Phase 3] Could not load validation cache; rescanning images: {exc}")
        valid_paths = []
        invalid_paths = []
        for sample in samples:
            path = sample[0]
            try:
                with Image.open(path) as image:
                    image.verify()
                self.samples.append(sample)
                valid_paths.append(path)
            except (OSError, ValueError):
                skipped += 1
                invalid_paths.append(path)
        if validation_cache_path:
            try:
                os.makedirs(os.path.dirname(validation_cache_path), exist_ok=True)
                torch.save({
                    "schema_version": 1,
                    "valid_paths": valid_paths,
                    "invalid_paths": invalid_paths,
                }, validation_cache_path)
                live_print(f"[Phase 3] Saved validation cache: {validation_cache_path} ({len(valid_paths)} valid, {len(invalid_paths)} invalid)")
            except Exception as exc:
                live_print(f"[Phase 3] Could not save validation cache: {exc}")
        if skipped:
            live_print(f"[Phase 3] Skipped {skipped} corrupt or unreadable labelled/pseudo samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.samples):
            raise IndexError(index)
        sample = self.samples[index]
        if len(sample) == 3:
            path, label, is_pseudo = sample
            box = None
            sample_weight = 1.0
        else:
            path, box, label, is_pseudo, sample_weight = sample
        try:
            image = Image.open(path).convert("RGB")
            if box is not None:
                width, height = image.size
                x1, y1, x2, y2 = box
                left = max(0, int(round(x1)))
                top = max(0, int(round(y1)))
                right = min(width, int(round(x2)))
                bottom = min(height, int(round(y2)))
                if right > left and bottom > top:
                    image = image.crop((left, top, right, bottom))
        except Exception as exc:
            raise RuntimeError(f"Could not decode Phase 3 image {path}: {exc}") from exc
        return self.transform(image), int(label), bool(is_pseudo), float(sample_weight)


def _initialize_phase3_loss_weights(
    cfg: SystemConfig,
    joint_loss_fn: BroadFaceCurricularLoss,
    num_labeled_classes: int,
    device: torch.device,
) -> None:
    """Copy trained Phase 1 classifier weights into joint_loss_fn so backbone isn't corrupted."""
    phase1_path = os.path.join(cfg.train.checkpoint_dir, "recognition_backbone.pt")
    if not os.path.isfile(phase1_path):
        return
    try:
        ckpt = torch.load(phase1_path, map_location=device, weights_only=False)
        loss_state = ckpt.get("broadface_loss_fn", {})
        saved_weight = loss_state.get("weight")
        if saved_weight is not None and saved_weight.ndim == 2:
            copy_rows = min(num_labeled_classes, saved_weight.shape[0], joint_loss_fn.weight.shape[0])
            with torch.no_grad():
                joint_loss_fn.weight[:copy_rows].copy_(saved_weight[:copy_rows].to(device))
            live_print(f"[Phase 3] Restored {copy_rows} Phase 1 class centers into joint_loss_fn")
    except Exception as exc:
        live_print(f"[Phase 3] Warning: could not copy Phase 1 classifier weights: {exc}")


def _finetune_phase3_joint(
    cfg: SystemConfig,
    raw_backbone: nn.Module,
    joint_dataset: Dataset,
    num_labeled_classes: int,
    total_classes: int,
    device: torch.device,
) -> BroadFaceCurricularLoss:
    """Warm up pseudo-class centers with frozen backbone, then gently adapt."""
    joint_loss_fn = BroadFaceCurricularLoss(
        in_features=cfg.model.embedding_dim,
        num_classes=total_classes,
        scale_factor=cfg.loss.scale,
        margin=cfg.loss.margin,
        alpha=cfg.loss.ema_alpha,
        queue_size=min(cfg.loss.queue_size, 4096),
        compensate=True,
        sync_curriculum=False,
    ).to(device)
    _initialize_phase3_loss_weights(cfg, joint_loss_fn, num_labeled_classes, device)

    if len(joint_dataset) < cfg.train.batch_size:
        raise ValueError(
            f"Adaptation dataset has {len(joint_dataset)} samples, fewer than batch_size={cfg.train.batch_size}"
        )
    joint_loader = DataLoader(
        joint_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )

    # Step A: 1 epoch warmup of ONLY the classifier head (backbone frozen in eval mode)
    raw_backbone.eval()
    for param in raw_backbone.parameters():
        param.requires_grad = False
    head_optimizer = optim.SGD(
        joint_loss_fn.parameters(),
        lr=cfg.train.learning_rate * 0.1,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
    )
    for images, labels, is_pseudo, sample_weights in joint_loader:
        images, labels = images.to(device), labels.to(device)
        is_pseudo = is_pseudo.bool().to(device)
        sample_weights = sample_weights.to(device)
        head_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            embeddings = raw_backbone(images)
        labelled_mask = ~is_pseudo
        pseudo_mask = is_pseudo
        total_loss = joint_loss_fn.weight.sum() * 0.0
        if labelled_mask.any():
            total_loss = total_loss + joint_loss_fn(embeddings[labelled_mask], labels[labelled_mask])
        if pseudo_mask.any():
            pseudo_loss = joint_loss_fn(embeddings[pseudo_mask], labels[pseudo_mask])
            total_loss = total_loss + cfg.loss.lambda_semi * sample_weights[pseudo_mask].mean() * pseudo_loss
        total_loss.backward()
        head_optimizer.step()

    # Step B: Unfreeze backbone with very small LR (0.01x) and frozen BN statistics
    for param in raw_backbone.parameters():
        param.requires_grad = True
    raw_backbone.eval()
    joint_optimizer = optim.SGD(
        [
            {"params": raw_backbone.parameters(), "lr": cfg.train.learning_rate * 0.01},
            {"params": joint_loss_fn.parameters(), "lr": cfg.train.learning_rate * 0.05},
        ],
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
    )
    for _ in range(min(2, cfg.train.epochs)):
        for images, labels, is_pseudo, sample_weights in joint_loader:
            images, labels = images.to(device), labels.to(device)
            is_pseudo = is_pseudo.bool().to(device)
            sample_weights = sample_weights.to(device)
            joint_optimizer.zero_grad(set_to_none=True)
            embeddings = raw_backbone(images)
            labelled_mask = ~is_pseudo
            pseudo_mask = is_pseudo
            total_loss = embeddings.sum() * 0.0
            if labelled_mask.any():
                total_loss = total_loss + joint_loss_fn(embeddings[labelled_mask], labels[labelled_mask])
            if pseudo_mask.any():
                pseudo_loss = joint_loss_fn(embeddings[pseudo_mask], labels[pseudo_mask])
                total_loss = total_loss + cfg.loss.lambda_semi * sample_weights[pseudo_mask].mean() * pseudo_loss
            total_loss.backward()
            joint_optimizer.step()
            joint_loss_fn.enqueue(embeddings.detach(), labels)
    raw_backbone.eval()
    return joint_loss_fn


def train_phase3_semi_supervised_gcn(cfg: SystemConfig, backbone, num_labeled_classes: int, device: torch.device, **kwargs):
    """Adapt recognition using explicit face crops or detector-approved scenes."""
    rank = kwargs.get("rank", 0)
    if not cfg.dataset.unlabeled_data_dir:
        if rank == 0:
            live_print("Phase 3 skipped: no unlabeled_data_dir configured.")
        return
    unlabeled_mode = cfg.dataset.unlabeled_mode
    if unlabeled_mode == "face_crops":
        dataset = UnlabeledFaceDataset(
            root_dir=cfg.dataset.unlabeled_data_dir,
            is_train=False,
            image_size=cfg.dataset.image_size,
        )
    elif unlabeled_mode == "detected_scenes":
        detector = kwargs.get("detector")
        if detector is None:
            raise FileNotFoundError(
                "Phase 3 detected_scenes mode requires a trained detector. "
                "Run Phase 0 first or use --unlabeled_mode face_crops."
            )
        dataset = DetectorCropDataset(
            root_dir=cfg.dataset.unlabeled_data_dir,
            detector=detector,
            device=device,
            score_threshold=cfg.dataset.unlabeled_score_threshold,
            image_size=cfg.dataset.image_size,
            cache_path=os.path.join(cfg.train.checkpoint_dir, "phase3_detector_crops.pt"),
        )
    else:
        raise ValueError(f"Unsupported unlabeled_mode: {unlabeled_mode}")
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.dataset.num_workers)
    raw_backbone = backbone.module if hasattr(backbone, "module") else backbone
    raw_backbone.eval()

    embedding_cache_path = os.path.join(cfg.train.checkpoint_dir, "phase3_embeddings.pt")
    detector_cache_path = os.path.join(cfg.train.checkpoint_dir, "phase3_detector_crops.pt")
    detector_cache_mtime = os.path.getmtime(detector_cache_path) if os.path.isfile(detector_cache_path) else 0.0

    all_embeddings = None
    if os.path.isfile(embedding_cache_path):
        try:
            cached = torch.load(embedding_cache_path, map_location="cpu", weights_only=False)
            if (cached.get("schema_version") == 1
                    and cached.get("detector_cache_mtime") == detector_cache_mtime
                    and cached.get("num_samples") == len(dataset)):
                all_embeddings = cached["embeddings"]
                if rank == 0:
                    live_print(f"[Phase 3] Loaded cached embeddings: {embedding_cache_path} ({all_embeddings.shape[0]} samples)")
        except Exception as exc:
            if rank == 0:
                live_print(f"[Phase 3] Could not load embedding cache; recomputing: {exc}")

    if all_embeddings is None:
        embeddings = []
        total_batches = max(1, len(loader))
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                images = batch[0]
                embeddings.append(raw_backbone(images.to(device)).cpu())
                if rank == 0 and (batch_idx == 1 or batch_idx % 500 == 0 or batch_idx == total_batches):
                    live_print(f"[Phase 3] Embedding batch {batch_idx}/{total_batches}")
        all_embeddings = torch.cat(embeddings, dim=0)
        if rank == 0:
            torch.save({
                "schema_version": 1,
                "detector_cache_mtime": detector_cache_mtime,
                "num_samples": len(dataset),
                "embeddings": all_embeddings,
            }, embedding_cache_path)
            live_print(f"[Phase 3] Saved cached embeddings: {embedding_cache_path} ({all_embeddings.shape[0]} samples)")

    pseudo_labels_cache_path = os.path.join(cfg.train.checkpoint_dir, "phase3_pseudo_labels.pt")
    labelled = build_identity_dataset(cfg, is_train=False)
    predictor = None

    if os.path.isfile(pseudo_labels_cache_path):
        try:
            cached = torch.load(pseudo_labels_cache_path, map_location="cpu", weights_only=False)
            if cached.get("schema_version") == 1 and cached.get("kind") == "phase3_pseudo_labels":
                pseudo_labels = cached["pseudo_labels"]
                active = cached["active_mask"]
                num_labeled_classes = cached["num_labeled_classes"]
                if rank == 0:
                    live_print(f"[Phase 3] Loaded cached pseudo-labels: {pseudo_labels_cache_path} ({int(active.sum().item())} active clusters)")
            else:
                pseudo_labels, active = None, None
        except Exception:
            pseudo_labels, active = None, None
    else:
        pseudo_labels, active = None, None

    if pseudo_labels is None or active is None:
        predictor = GCNLinkPredictor(feature_dim=cfg.model.embedding_dim, hidden_dim=256, k_neighbors=5).to(device)
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    active_indices = torch.where(active.cpu())[0].tolist()
    pseudo_samples = []
    for index in active_indices:
        if unlabeled_mode == "detected_scenes":
            path, box, score = dataset.samples[index]
            pseudo_samples.append((path, box, int(pseudo_labels[index].item()), True, score))
        elif dataset.samples[index] != "dummy_unlabeled":
            pseudo_samples.append((dataset.samples[index], int(pseudo_labels[index].item()), True))
    labelled_samples = []
    for path, box, label in labelled.samples:
        pixel_box = None
        if box is not None:
            try:
                with Image.open(path) as image:
                    image_width, image_height = image.size
                xc, yc, box_width, box_height = box
                pixel_box = (
                    max(0, int((xc - box_width / 2.0) * image_width)),
                    max(0, int((yc - box_height / 2.0) * image_height)),
                    min(image_width, int((xc + box_width / 2.0) * image_width)),
                    min(image_height, int((yc + box_height / 2.0) * image_height)),
                )
            except Exception as exc:
                raise RuntimeError(f"Could not prepare WIDER adaptation crop {path}: {exc}") from exc
        labelled_samples.append((path, pixel_box, label, False, 1.0))

    joint_dataset = _CombinedIdentityDataset(
        labelled_samples + pseudo_samples,
        cfg.dataset.image_size,
        validation_cache_path=os.path.join(cfg.train.checkpoint_dir, "phase3_image_validation.pt"),
    )
    joint_loss_fn = None
    if len(joint_dataset) and pseudo_samples:
        pseudo_class_ids = [
            sample[2] if len(sample) == 5 else sample[1]
            for sample in pseudo_samples
        ]
        total_classes = max(num_labeled_classes, max(pseudo_class_ids) + 1)
        if rank == 0:
            torch.save({
                "schema_version": 1,
                "kind": "phase3_pseudo_labels",
                "pseudo_labels": pseudo_labels.cpu(),
                "active_mask": active.cpu(),
                "num_labeled_classes": num_labeled_classes,
                "total_classes": total_classes,
            }, pseudo_labels_cache_path)
        joint_loss_fn = _finetune_phase3_joint(
            cfg, raw_backbone, joint_dataset, num_labeled_classes, total_classes, device
        )

    checkpoint_path = os.path.join(cfg.train.checkpoint_dir, "phase3_semi_gcn_predictor.pt")
    if rank == 0:
        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
        torch.save({
            "schema_version": 2,
            "kind": "gcn_adapter",
            "gcn_predictor": predictor.state_dict() if predictor is not None else None,
            "pseudo_labels": pseudo_labels.cpu(),
            "active_mask": active.cpu(),
            "model_config": asdict(cfg.model),
            "input_size": list(cfg.dataset.image_size),
            "recognition_alignment": cfg.model.recognition_alignment,
        }, checkpoint_path)
        torch.save({
            "schema_version": 2,
            "kind": "recognition_backbone",
            "backbone": raw_backbone.state_dict(),
            "backbone_state": raw_backbone.state_dict(),
            "broadface_loss_fn": joint_loss_fn.state_dict() if joint_loss_fn is not None else None,
            "epoch": min(2, cfg.train.epochs),
            "model_config": asdict(cfg.model),
            "input_size": list(cfg.dataset.image_size),
            "loss_config": asdict(cfg.loss),
            "class_to_idx": dict(labelled.class_to_idx),
            "recognition_alignment": cfg.model.recognition_alignment,
        }, os.path.join(cfg.train.checkpoint_dir, "phase3_adapted_backbone.pt"))
        live_print(f"Phase 3 generated {int(active.sum())} globally indexed pseudo-labelled samples; saved {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="OccuPose-BroadDictNet training")
    parser.add_argument(
        "--phase",
        choices=["all", "detector", "recognition", "adapt", "enroll"],
        default="all",
        help="Run detector pretraining, identity recognition, unlabeled adaptation, or the configured sequence",
    )
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--bb_dir", default=None, help="Path to AQUA bb_label YOLO/Darknet detection dataset")
    parser.add_argument("--unlabeled_dir", default=None)
    parser.add_argument("--wider_dir", default=None, help="Optional nested WIDER folder-label dataset under unlabeled/")
    parser.add_argument("--celeba_dir", default=None)
    parser.add_argument("--backbone", choices=["iresnet100", "vit_face_base", "fasterrcnn_backbone"], default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--detector_batch_size", type=int, default=None)
    parser.add_argument("--detector_epochs", type=int, default=None)
    parser.add_argument("--detector_min_size", type=int, default=None)
    parser.add_argument("--detector_max_size", type=int, default=None)
    parser.add_argument("--detector_split", choices=["all", "train"], default=None)
    parser.add_argument("--unlabeled_mode", choices=["detected_scenes", "face_crops"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retrain_detector", action="store_true")
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
    if args.wider_dir is not None: cfg.dataset.wider_identity_data_dir = args.wider_dir
    if args.celeba_dir is not None: cfg.dataset.celeba_dir = args.celeba_dir
    if args.backbone is not None: cfg.model.backbone_type = args.backbone
    if args.checkpoint_dir is not None: cfg.train.checkpoint_dir = args.checkpoint_dir
    if args.epochs is not None: cfg.train.epochs = args.epochs
    if args.batch_size is not None: cfg.train.batch_size = args.batch_size
    if args.detector_batch_size is not None: cfg.train.detector_batch_size = args.detector_batch_size
    if args.detector_epochs is not None: cfg.train.detector_epochs = args.detector_epochs
    if args.detector_min_size is not None: cfg.train.detector_min_size = args.detector_min_size
    if args.detector_max_size is not None: cfg.train.detector_max_size = args.detector_max_size
    if args.detector_split is not None: cfg.train.detector_split = args.detector_split
    if args.unlabeled_mode is not None: cfg.dataset.unlabeled_mode = args.unlabeled_mode
    if args.lr is not None: cfg.train.learning_rate = args.lr
    cfg.model.fp16 = args.fp16 or (device.type == "cuda")
    cfg.dataset.allow_synthetic_fixtures = args.allow_synthetic_fixtures
    seed_everything(cfg.train.seed + rank)

    detector_checkpoint = None
    if args.phase in {"all", "detector"}:
        if args.phase == "detector" and not (
            cfg.dataset.bb_data_dir or find_wider_special_root(cfg.dataset.unlabeled_data_dir)
        ):
            raise ValueError("--phase detector requires --bb_dir or a nested WIDER dataset under --unlabeled_dir")
        detector_checkpoint = train_phase0_face_detector(
            cfg,
            device,
            is_ddp=is_ddp,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            force_retrain=args.retrain_detector or args.retrain_all,
        )
        if args.phase == "detector":
            if is_ddp and dist.is_initialized():
                dist.destroy_process_group()
            return

    backbone = None
    num_classes = 0
    if args.phase in {"all", "recognition"}:
        if not cfg.dataset.train_data_dir:
            if args.phase == "recognition":
                raise ValueError("--phase recognition requires --data_dir pointing to name_label")
            if rank == 0:
                live_print("Recognition phase skipped: no data_dir configured.")
        else:
            backbone, num_classes = train_phase1_backbone_curricular(
                cfg, device, is_ddp=is_ddp, rank=rank, local_rank=local_rank,
                world_size=world_size, resume=args.resume,
                force_retrain=args.retrain_phase1 or args.retrain_all,
            )
            if cfg.dataset.celeba_dir:
                train_phase2_anet_attributes(
                    cfg, backbone, device, rank=rank,
                    force_retrain=args.retrain_phase2 or args.retrain_all,
                )
                if is_ddp and dist.is_initialized():
                    dist.barrier()
                if rank == 0:
                    live_print("Phase 2 complete; proceeding to Phase 3 adaptation.")
            elif rank == 0:
                live_print("Phase 2 skipped: no celeba_dir configured.")

    if args.phase == "enroll":
        if not cfg.dataset.train_data_dir:
            raise ValueError("--phase enroll requires --data_dir to recover the identity class map")
        backbone, _ = load_backbone_for_adaptation(cfg, device)
        enroll_final_gallery(cfg, backbone, device)
        return

    if args.phase == "adapt":
        if not cfg.dataset.train_data_dir:
            raise ValueError("--phase adapt requires --data_dir to recover the identity class map")
        if not cfg.dataset.unlabeled_data_dir:
            raise ValueError("--phase adapt requires --unlabeled_dir")
        backbone, num_classes = load_backbone_for_adaptation(cfg, device)

    if args.phase in {"all", "adapt"} and cfg.dataset.unlabeled_data_dir:
        if rank == 0:
            live_print(
                "Phase 3 starting: building or loading detector-approved crop cache "
                f"under {cfg.train.checkpoint_dir}"
            )
        if backbone is None:
            raise ValueError("Unlabeled adaptation requires a trained recognition backbone")
        runtime_detector = None
        if cfg.dataset.unlabeled_mode == "detected_scenes":
            detector_path = detector_checkpoint or os.path.join(cfg.train.checkpoint_dir, "face_detector.pt")
            if not os.path.isfile(detector_path):
                raise FileNotFoundError(
                    f"Detected-scene adaptation requires a face detector checkpoint: {detector_path}"
                )
            runtime_detector = TorchvisionFaceDetector(
                checkpoint_path=detector_path,
                pretrained_backbone=False,
            ).to(device).eval()
        train_phase3_semi_supervised_gcn(
            cfg,
            backbone,
            num_classes,
            device,
            rank=rank,
            detector=runtime_detector,
        )
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0 and backbone is not None:
        enroll_final_gallery(cfg, backbone, device)


if __name__ == "__main__":
    main()
