import os
import re
import glob
from PIL import Image
import torch
import torch.utils.data as data

try:
    import torchvision.transforms as transforms
except Exception:
    transforms = None

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

class PurePyTorchImageTransform:
    """
    Fallback Image Transform using pure PyTorch & PIL.
    Requires ZERO external C-libraries.
    """
    def __init__(self, image_size=(112, 112), is_train=True):
        self.image_size = image_size
        self.is_train = is_train

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.size != self.image_size:
            img = img.resize(self.image_size, Image.BILINEAR)
            
        if self.is_train and torch.rand(1).item() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
        img_bytes = img.tobytes()
        tensor = torch.frombuffer(img_bytes, dtype=torch.uint8)
        tensor = tensor.view(self.image_size[1], self.image_size[0], 3).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor

def get_default_transform(image_size=(112, 112), is_train=True):
    if transforms is not None:
        try:
            if is_train:
                return transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        except Exception:
            pass
    return PurePyTorchImageTransform(image_size=image_size, is_train=is_train)

def parse_yolo_txt(txt_path: str):
    """
    Parses YOLO format bounding box annotation file (.txt).
    """
    bboxes = []
    try:
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            vals = [float(x) for x in parts[1:5]]
                            bboxes.append(vals)
                        except ValueError:
                            pass
    except Exception:
        pass
    return bboxes

class UnlabeledWildDataset(data.Dataset):
    """
    Dynamic Dataset Loader for 'unlabeled/' directory.
    Supports dynamic rescanning of constantly growing datasets uploaded in parallel.
    """
    def __init__(self, unlabeled_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112)):
        super(UnlabeledWildDataset, self).__init__()
        self.unlabeled_dir = unlabeled_dir
        self.image_size = image_size
        self.transform = transform if transform is not None else get_default_transform(image_size, is_train)
        self.samples = []
        self.rescan()

    def rescan(self):
        """
        Dynamically rescans the unlabeled directory to index newly arrived images uploaded in parallel.
        """
        prev_count = len(self.samples)
        self.samples = []
        if self.unlabeled_dir and os.path.exists(self.unlabeled_dir):
            for root, _, files in os.walk(self.unlabeled_dir):
                for f in files:
                    if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                        self.samples.append(os.path.join(root, f))
        else:
            for _ in range(500):
                self.samples.append("dummy_unlabeled")
                
        if len(self.samples) != prev_count:
            print(f"[Dynamic Unlabeled Rescan] Indexed {len(self.samples)} images (New: +{len(self.samples) - prev_count})")
        return len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx % len(self.samples)]
        if path == "dummy_unlabeled":
            img_tensor = torch.randn(3, *self.image_size)
        else:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
            except Exception:
                img_tensor = torch.zeros(3, *self.image_size)
                
        return img_tensor, -1, False


class NameLabeledFaceDataset(data.Dataset):
    """
    Dynamic Dataset Loader for 'name_label/' directory.
    Supports dynamic rescanning of constantly growing labeled datasets uploaded in parallel.
    """
    def __init__(self, name_label_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112)):
        super(NameLabeledFaceDataset, self).__init__()
        self.name_label_dir = name_label_dir
        self.image_size = image_size
        self.transform = transform if transform is not None else get_default_transform(image_size, is_train)
        self.samples = []
        self.class_to_idx = {}
        self.rescan()

    def rescan(self):
        """
        Dynamically rescans name_label/ directory to index newly arrived identity images.
        """
        prev_count = len(self.samples)
        self.samples = []
        
        if self.name_label_dir and os.path.exists(self.name_label_dir):
            self._parse_name_label_directory()
        else:
            self.class_to_idx = {f"Subject_{i}": i for i in range(50)}
            for i in range(500):
                self.samples.append(("dummy_labeled", i % 50))
                
        if len(self.samples) != prev_count:
            print(f"[Dynamic NameLabeled Rescan] Indexed {len(self.samples)} images across {len(self.class_to_idx)} identities (New: +{len(self.samples) - prev_count})")
        return len(self.samples)

    def _parse_name_label_directory(self):
        rof_dir = os.path.join(self.name_label_dir, "ROF")
        if os.path.exists(rof_dir):
            self._parse_folder_identities(rof_dir, prefix="ROF_")
            
        wild_dir = os.path.join(self.name_label_dir, "face_detection_in_wild_dataset")
        if os.path.exists(wild_dir):
            self._parse_folder_identities(wild_dir, prefix="WILD_")
            
        mask_dir = os.path.join(self.name_label_dir, "face_with_mask")
        if os.path.exists(mask_dir):
            self._parse_filename_identities(mask_dir)
            
        if len(self.samples) == 0:
            self._parse_folder_identities(self.name_label_dir, prefix="")

    def _parse_folder_identities(self, base_dir: str, prefix: str = ""):
        for root, dirs, files in os.walk(base_dir):
            image_files = [f for f in files if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
            if len(image_files) > 0:
                folder_name = prefix + os.path.basename(root)
                if folder_name not in self.class_to_idx:
                    self.class_to_idx[folder_name] = len(self.class_to_idx)
                label_idx = self.class_to_idx[folder_name]
                
                for f in image_files:
                    self.samples.append((os.path.join(root, f), label_idx))

    def _parse_filename_identities(self, base_dir: str):
        pattern = re.compile(r'^(.*?)(?:_\d+)?\.(?:jpg|jpeg|png|bmp|webp)$', re.IGNORECASE)
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    match = pattern.match(f)
                    raw_name = match.group(1) if match else "unknown_mask"
                    identity_name = "MASK_" + raw_name
                    
                    if identity_name not in self.class_to_idx:
                        self.class_to_idx[identity_name] = len(self.class_to_idx)
                    label_idx = self.class_to_idx[identity_name]
                    self.samples.append((os.path.join(root, f), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx % len(self.samples)]
        if path == "dummy_labeled":
            img_tensor = torch.randn(3, *self.image_size)
        else:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
            except Exception:
                img_tensor = torch.zeros(3, *self.image_size)
                
        return img_tensor, label_idx, True


class BoundingBoxFaceDataset(data.Dataset):
    """
    Dynamic Dataset Loader for 'bb_label/' directory.
    Supports dynamic rescanning of bounding box datasets uploaded in parallel.
    """
    def __init__(self, bb_label_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112)):
        super(BoundingBoxFaceDataset, self).__init__()
        self.bb_label_dir = bb_label_dir
        self.image_size = image_size
        self.transform = transform if transform is not None else get_default_transform(image_size, is_train)
        self.samples = []
        self.rescan()

    def rescan(self):
        """
        Dynamically rescans bb_label/ directory for newly uploaded YOLO / Darknet bounding box images.
        """
        prev_count = len(self.samples)
        self.samples = []
        if self.bb_label_dir and os.path.exists(self.bb_label_dir):
            self._parse_bb_label_directory()
        else:
            for i in range(200):
                self.samples.append(("dummy_bb", None, 0))
                
        if len(self.samples) != prev_count:
            print(f"[Dynamic BoundingBox Rescan] Indexed {len(self.samples)} bounding box images (New: +{len(self.samples) - prev_count})")
        return len(self.samples)

    def _parse_bb_label_directory(self):
        for root, _, files in os.walk(self.bb_label_dir):
            for f in files:
                if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    img_path = os.path.join(root, f)
                    base_name = os.path.splitext(f)[0]
                    
                    txt_path = os.path.join(root, base_name + ".txt")
                    if not os.path.exists(txt_path) and "/images" in root:
                        labels_root = root.replace("/images", "/labels")
                        txt_path = os.path.join(labels_root, base_name + ".txt")
                        
                    bboxes = parse_yolo_txt(txt_path)
                    self.samples.append((img_path, bboxes, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, bboxes, class_id = self.samples[idx % len(self.samples)]
        if path == "dummy_bb":
            img_tensor = torch.randn(3, *self.image_size)
            return img_tensor, class_id, True
            
        try:
            img = Image.open(path).convert('RGB')
            w_img, h_img = img.size
            
            if bboxes and len(bboxes) > 0:
                xc, yc, w_box, h_box = bboxes[0]
                x1 = max(0, int((xc - w_box / 2.0) * w_img))
                y1 = max(0, int((yc - h_box / 2.0) * h_img))
                x2 = min(w_img, int((xc + w_box / 2.0) * w_img))
                y2 = min(h_img, int((yc + h_box / 2.0) * h_img))
                
                if x2 > x1 + 5 and y2 > y1 + 5:
                    img = img.crop((x1, y1, x2, y2))
                    
            img_tensor = self.transform(img)
        except Exception:
            img_tensor = torch.zeros(3, *self.image_size)
            
        return img_tensor, class_id, True


class UnifiedWildFaceDataset(data.Dataset):
    """
    Unified Dynamic Dataset Loader combining all datastreams with automatic epoch-wise rescanning.
    """
    def __init__(self, root_dir: str = None, meta_file: str = None, is_labeled: bool = True, transform=None, is_train: bool = True, image_size=(112, 112)):
        super(UnifiedWildFaceDataset, self).__init__()
        self.root_dir = root_dir
        self.is_labeled = is_labeled
        self.is_train = is_train
        
        self.unlabeled_ds = UnlabeledWildDataset(
            unlabeled_dir=os.path.join(root_dir, "unlabeled") if root_dir else None,
            transform=transform, is_train=is_train, image_size=image_size
        )
        
        self.name_labeled_ds = NameLabeledFaceDataset(
            name_label_dir=os.path.join(root_dir, "name_label") if root_dir else None,
            transform=transform, is_train=is_train, image_size=image_size
        )
        
        self.bb_labeled_ds = BoundingBoxFaceDataset(
            bb_label_dir=os.path.join(root_dir, "bb_label") if root_dir else None,
            transform=transform, is_train=is_train, image_size=image_size
        )
        
        self.class_to_idx = self.name_labeled_ds.class_to_idx

    def rescan(self):
        """
        Rescans all sub-datasets to pick up newly arrived parallel file uploads.
        """
        n_unlabeled = self.unlabeled_ds.rescan()
        n_name = self.name_labeled_ds.rescan()
        n_bb = self.bb_labeled_ds.rescan()
        self.class_to_idx = self.name_labeled_ds.class_to_idx
        return n_unlabeled + n_name + n_bb

    def __len__(self):
        return len(self.unlabeled_ds) + len(self.name_labeled_ds) + len(self.bb_labeled_ds)

    def __getitem__(self, idx):
        if self.root_dir is None:
            if self.is_labeled:
                return self.name_labeled_ds[idx % len(self.name_labeled_ds)]
            else:
                return self.unlabeled_ds[idx % len(self.unlabeled_ds)]
            
        len_unlabeled = len(self.unlabeled_ds) if self.root_dir else 0
        len_name = len(self.name_labeled_ds)
        
        if idx < len_unlabeled:
            return self.unlabeled_ds[idx]
        elif idx < len_unlabeled + len_name:
            return self.name_labeled_ds[idx - len_unlabeled]
        else:
            return self.bb_labeled_ds[idx - len_unlabeled - len_name]


RobustUniversalFaceDataset = UnifiedWildFaceDataset
WildFaceDataset = NameLabeledFaceDataset
UnlabeledFaceDataset = UnlabeledWildDataset

class CelebAAttributeDataset(data.Dataset):
    def __init__(self, root_dir: str = None, attr_file: str = None, is_train: bool = True, image_size=(112, 112)):
        super(CelebAAttributeDataset, self).__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = get_default_transform(image_size, is_train)
        
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
        path, attr_targets = self.samples[idx % len(self.samples)]
        if path == "dummy_attr":
            img_tensor = torch.randn(3, 112, 112)
        else:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
            except Exception:
                img_tensor = torch.zeros(3, 112, 112)
            
        return img_tensor, attr_targets
