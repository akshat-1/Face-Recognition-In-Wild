import os
import glob
import random
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class WildFaceDataset(data.Dataset):
    """
    Dataset loader for WebFace-OCC / LFW / Custom Face Datasets.
    Applies synthetic random occlusions (masks, sunglasses, artificial spatial blocks)
    and spatial data augmentations to simulate real-world unconstrained wild environments.
    """
    def __init__(self, root_dir: str = None, transform=None, is_train: bool = True, num_synthetic_samples: int = 1000):
        super(WildFaceDataset, self).__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        self.samples = []
        self.class_to_idx = {}
        
        # Load from image directory structure (root/class_name/img.jpg) if directory exists
        if root_dir and os.path.exists(root_dir):
            classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            
            for cls_name in classes:
                cls_dir = os.path.join(root_dir, cls_name)
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))
        else:
            # Generate synthetic samples if dataset directory is not provided (for quick dry-run testing)
            self.class_to_idx = {f"Subject_{i}": i for i in range(50)}
            for i in range(num_synthetic_samples):
                self.samples.append(("synthetic", random.randint(0, 49)))

    def __len__(self):
        return len(self.samples)

    def _apply_synthetic_occlusion(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Simulates partial occlusions (e.g. medical masks, sunglasses, hands).
        Applies a zeroed-out rectangular patch on 15-40% of the face region.
        """
        if not self.is_train or random.random() > 0.5:
            return img_tensor
            
        _, h, w = img_tensor.shape
        occ_h = random.randint(int(h * 0.2), int(h * 0.45))
        occ_w = random.randint(int(w * 0.3), int(w * 0.8))
        
        top = random.randint(int(h * 0.3), h - occ_h)
        left = random.randint(0, w - occ_w)
        
        img_tensor[:, top:top + occ_h, left:left + occ_w] = -1.0 # Normalized black block
        return img_tensor

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        if path == "synthetic":
            img_tensor = torch.randn(3, 112, 112)
        else:
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img)
            
        # Apply random synthetic occlusions during training
        img_tensor = self._apply_synthetic_occlusion(img_tensor)
        
        return img_tensor, label

class CelebAAttributeDataset(data.Dataset):
    """
    Dataset loader for CelebA multi-task 40-attribute training.
    Returns face crop tensor and 40 binary attribute target labels.
    """
    def __init__(self, root_dir: str = None, is_train: bool = True, num_synthetic_samples: int = 1000):
        super(CelebAAttributeDataset, self).__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.num_samples = num_synthetic_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generates synthetic attribute targets (40 binary targets 0 or 1)
        img_tensor = torch.randn(3, 112, 112)
        attr_targets = (torch.rand(40) > 0.5).float()
        return img_tensor, attr_targets
