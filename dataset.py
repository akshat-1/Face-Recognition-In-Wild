import os
import glob
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class WildFaceDataset(data.Dataset):
    """
    Production Dataset loader for large-scale wild face datasets containing real occlusions
    (e.g., WebFace-OCC, LFW, MS1MV2, CASIA-WebFace).
    
    Reads real unconstrained wild face images directly without adding artificial occlusions.
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
        
        # Load from image directory structure: root/class_name/img.jpg
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
            # Fallback synthetic tensor generator for testing when no path is supplied
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
            
        return img_tensor, label

class CelebAAttributeDataset(data.Dataset):
    """
    Dataset loader for CelebA multi-task 40-attribute training.
    Returns 112x112 face crop tensor and 40 binary attribute ground-truth targets.
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
                lines = f.readlines()[2:] # Skip header
                for line in lines:
                    parts = line.strip().split()
                    img_name = parts[0]
                    # Convert -1/1 attribute values to 0/1 binary targets
                    attrs = [1.0 if int(x) == 1 else 0.0 for x in parts[1:]]
                    self.samples.append((os.path.join(root_dir, img_name), torch.tensor(attrs, dtype=torch.float32)))
        else:
            # Fallback synthetic attribute samples for dry-run testing
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
