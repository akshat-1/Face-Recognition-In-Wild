import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class DatasetConfig:
    train_data_dir: str = ""
    val_data_dir: str = ""
    celeba_dir: str = ""
    image_size: Tuple[int, int] = (112, 112)
    num_classes: int = 10575 # Default WebFace-OCC number of identities
    num_workers: int = 4
    use_synthetic_occlusions: bool = False # Real wild dataset already contains real occlusions

@dataclass
class ModelConfig:
    backbone_type: str = "iresnet100" # SOTA Improved ResNet-100 (3, 13, 30, 3)
    embedding_dim: int = 512
    fp16: bool = True
    use_pim_frontalizer: bool = True
    use_anet_attributes: bool = True
    lista_num_layers: int = 5
    atoms_per_class: int = 20

@dataclass
class LossConfig:
    scale: float = 64.0
    margin: float = 0.50
    ema_alpha: float = 0.99
    queue_size: int = 32768
    broadface_momentum: float = 0.99

@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: int = 2
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 50
    device: str = "cuda"

@dataclass
class SystemConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
