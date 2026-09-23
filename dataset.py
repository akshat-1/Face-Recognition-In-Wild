import os
import glob
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class WildFaceDataset(data.Dataset):
    """
    Production Dataset loader for labeled face datasets containing real occlusions
    (e.g., ROF masked/sunglasses, LFW, WIDER FACE, WebFace-OCC).
    
    Returns:
        img_tensor: (3, 112, 112) normalized image tensor
        label: integer identity class index >= 0
        is_labeled: True boolean flag
    """
    def __init__(self, root_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112)):
        super(WildFaceDataset, self).__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        
        if transform is None:
            if is_train:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        else:
            self.transform = transform
            
        self.samples = []
        self.class_to_idx = {}
        
        # Load from directory structure: root/class_name/img.jpg
        if root_dir and os.path.exists(root_dir):
            classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            for cls_name in classes:
                cls_dir = os.path.join(root_dir, cls_name)
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(valid_extensions):
                        self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))
        else:
            # Fallback synthetic dataset for testing when no directory is provided
            self.class_to_idx = {f"Subject_{i}": i for i in range(50)}
            for i in range(500):
                self.samples.append(("dummy_tensor", i % 50))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        if path == "dummy_tensor":
            img_tensor = torch.randn(3, 112, 112)
        else:
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img)
            
        return img_tensor, label, True # is_labeled = True

class UnlabeledFaceDataset(data.Dataset):
    """
    Production Dataset loader for unannotated wild face collections containing millions of images
    (e.g., FMD dataset, COVID face detection, unlabelled web crawls).
    
    Returns:
        img_tensor: (3, 112, 112) normalized image tensor
        label: -1 (unlabeled indicator)
        is_labeled: False boolean flag
    """
    def __init__(self, root_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112)):
        super(UnlabeledFaceDataset, self).__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        self.samples = []
        if root_dir and os.path.exists(root_dir):
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        self.samples.append(os.path.join(root, file))
        else:
            # Fallback dummy samples for testing
            for _ in range(500):
                self.samples.append("dummy_unlabeled")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        if path == "dummy_unlabeled":
            img_tensor = torch.randn(3, 112, 112)
        else:
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img)
            
        return img_tensor, -1, False # is_labeled = False

class CelebAAttributeDataset(data.Dataset):
    """
    Dataset loader for CelebA multi-task 40-attribute training.
    """
    def __init__(self, root_dir: str = None, attr_file: str = None, is_train: bool = True, image_size=(112, 112)):
        super(CelebAAttributeDataset, self).__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.samples = []
        if root_dir and attr_file and os.path.exists(attr_file):
            with open(attr_file, 'r') as f:
                lines = f.readlines()[2:]
                for line in lines:
                    parts = line.strip().split()
                    img_name = parts[0]
                    attrs = [1.0 if int(x) == 1 else 0.0 for x in parts[1:]]
                    self.samples.append((os.path.join(root_dir, img_name), torch.tensor(attrs, dtype=torch.float32)))
        else:
            for i in range(200):
                self.samples.append(("dummy_attr", (torch.rand(40) > 0.5).float()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, attr_targets = self.samples[idx]
        if path == "dummy_attr":
            img_tensor = torch.randn(3, 112, 112)
        else:
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img)
            
        return img_tensor, attr_targets
