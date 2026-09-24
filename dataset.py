import os
import re
import glob
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def parse_yolo_txt(txt_path: str):
    """
    Parses YOLO format bounding box annotation file (.txt).
    Format per line: class_id x_center y_center width height (normalized [0, 1])
    Returns list of bboxes: [[xc, yc, w, h], ...]
    """
    bboxes = []
    try:
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        vals = [float(x) for x in parts[1:5]]
                        bboxes.append(vals) # [xc, yc, w, h]
    except Exception:
        pass
    return bboxes

class UnlabeledWildDataset(data.Dataset):
    """
    Dataset Loader for 'unlabeled/' directory.
    Recursively discovers all face images across arbitrary subfolder depths and locations.
    Returns 3-tuple: (img_tensor, -1, False)
    """
    def __init__(self, unlabeled_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112)):
        super(UnlabeledWildDataset, self).__init__()
        self.unlabeled_dir = unlabeled_dir
        self.image_size = image_size
        
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
        if unlabeled_dir and os.path.exists(unlabeled_dir):
            for root, _, files in os.walk(unlabeled_dir):
                for f in files:
                    if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                        self.samples.append(os.path.join(root, f))
        else:
            # Fallback synthetic samples for dry-run testing
            for _ in range(500):
                self.samples.append("dummy_unlabeled")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        if path == "dummy_unlabeled":
            img_tensor = torch.randn(3, *self.image_size)
        else:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
            except Exception:
                img_tensor = torch.zeros(3, *self.image_size)
                
        return img_tensor, -1, False # is_labeled = False


class NameLabeledFaceDataset(data.Dataset):
    """
    Dataset Loader for 'name_label/' directory containing:
    1. ROF/ : Subfolders are identity names (e.g. ROF/Person_A/img1.jpg -> "Person_A")
    2. face_with_mask/ : Filenames contain identity names at various folder depths (e.g. Elon_Musk_0001.jpg -> "Elon_Musk")
    3. face_detection_in_wild_dataset/ : Subfolders are identity names (e.g. face_detection_in_wild_dataset/Subject_B/img2.jpg -> "Subject_B")
    
    Returns 3-tuple: (img_tensor, label_idx, True)
    """
    def __init__(self, name_label_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112)):
        super(NameLabeledFaceDataset, self).__init__()
        self.name_label_dir = name_label_dir
        self.image_size = image_size
        
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
        
        if name_label_dir and os.path.exists(name_label_dir):
            self._parse_name_label_directory()
        else:
            # Fallback synthetic labeled data for dry-run testing
            self.class_to_idx = {f"Subject_{i}": i for i in range(50)}
            for i in range(500):
                self.samples.append(("dummy_labeled", i % 50))

    def _parse_name_label_directory(self):
        # 1. Parse ROF/ (subfolders = identities)
        rof_dir = os.path.join(self.name_label_dir, "ROF")
        if os.path.exists(rof_dir):
            self._parse_folder_identities(rof_dir, prefix="ROF_")
            
        # 2. Parse face_detection_in_wild_dataset/ (subfolders = identities)
        wild_dir = os.path.join(self.name_label_dir, "face_detection_in_wild_dataset")
        if os.path.exists(wild_dir):
            self._parse_folder_identities(wild_dir, prefix="WILD_")
            
        # 3. Parse face_with_mask/ (filenames contain identity names at various depths)
        mask_dir = os.path.join(self.name_label_dir, "face_with_mask")
        if os.path.exists(mask_dir):
            self._parse_filename_identities(mask_dir)
            
        # Fallback if specific subfolders aren't named exactly, walk whole name_label_dir
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
        # Regex to extract person identity name from filenames like Elon_Musk_0001.jpg or subject42_masked.png
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
        path, label_idx = self.samples[idx]
        if path == "dummy_labeled":
            img_tensor = torch.randn(3, *self.image_size)
        else:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
            except Exception:
                img_tensor = torch.zeros(3, *self.image_size)
                
        return img_tensor, label_idx, True # is_labeled = True


class BoundingBoxFaceDataset(data.Dataset):
    """
    Dataset Loader for 'bb_label/' directory containing:
    1. facemaskyolo/data/ : Images in 'images/' folder, labels in 'labels/' folder (YOLO .txt files)
    2. darknet/ : Images and label .txt files in the SAME folder with matching base names
    
    Crops face region using YOLO bounding box coordinates before returning.
    Returns 3-tuple: (cropped_face_tensor, label_idx, True)
    """
    def __init__(self, bb_label_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112)):
        super(BoundingBoxFaceDataset, self).__init__()
        self.bb_label_dir = bb_label_dir
        self.image_size = image_size
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        self.samples = [] # List of tuples: (image_path, bbox_list, class_id)
        
        if bb_label_dir and os.path.exists(bb_label_dir):
            self._parse_bb_label_directory()
        else:
            # Fallback synthetic samples
            for i in range(200):
                self.samples.append(("dummy_bb", None, 0))

    def _parse_bb_label_directory(self):
        # 1. Parse facemaskyolo/data/ (images/ and labels/ separate subfolders)
        yolo_images_dir = os.path.join(self.bb_label_dir, "facemaskyolo", "data", "images")
        yolo_labels_dir = os.path.join(self.bb_label_dir, "facemaskyolo", "data", "labels")
        
        if os.path.exists(yolo_images_dir):
            for img_name in os.listdir(yolo_images_dir):
                if img_name.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    img_path = os.path.join(yolo_images_dir, img_name)
                    base_name = os.path.splitext(img_name)[0]
                    txt_path = os.path.join(yolo_labels_dir, base_name + ".txt") if os.path.exists(yolo_labels_dir) else ""
                    
                    bboxes = parse_yolo_txt(txt_path)
                    self.samples.append((img_path, bboxes, 0))
                    
        # 2. Parse darknet/ (images and label .txt files in the SAME folder)
        darknet_dir = os.path.join(self.bb_label_dir, "darknet")
        if os.path.exists(darknet_dir):
            for root, _, files in os.walk(darknet_dir):
                for f in files:
                    if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                        img_path = os.path.join(root, f)
                        base_name = os.path.splitext(f)[0]
                        txt_path = os.path.join(root, base_name + ".txt")
                        
                        bboxes = parse_yolo_txt(txt_path)
                        self.samples.append((img_path, bboxes, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, bboxes, class_id = self.samples[idx]
        
        if path == "dummy_bb":
            img_tensor = torch.randn(3, *self.image_size)
            return img_tensor, class_id, True
            
        try:
            img = Image.open(path).convert('RGB')
            w_img, h_img = img.size
            
            # Crop using first valid bounding box if available
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
    Unified Production Dataset Loader combining:
    - unlabeled/ (UnlabeledWildDataset)
    - name_label/ (NameLabeledFaceDataset: ROF, face_with_mask, face_detection_in_wild_dataset)
    - bb_label/ (BoundingBoxFaceDataset: facemaskyolo, darknet)
    
    Automatically routes images and returns (img_tensor, label_idx, is_labeled).
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


# Backwards compatibility Aliases across codebase
RobustUniversalFaceDataset = UnifiedWildFaceDataset
WildFaceDataset = NameLabeledFaceDataset
UnlabeledFaceDataset = UnlabeledWildDataset

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
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
            except Exception:
                img_tensor = torch.zeros(3, 112, 112)
            
        return img_tensor, attr_targets
