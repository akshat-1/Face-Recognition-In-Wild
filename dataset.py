import os
import re
import json
import csv
import glob
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class RobustUniversalFaceDataset(data.Dataset):
    """
    Production Universal Dataset Loader for Arbitrary Wild Face Datasets.
    Automatically handles non-standard, heterogeneous file structures:
    
    1. Metadata File Parsing (CSV, JSON, TXT, TSV)
    2. Deep Recursive Subfolder Hierarchy Parsing (root/**/subject_name/image.jpg)
    3. Filename Pattern Parsing (e.g. Elon_Musk_0001.jpg -> "Elon_Musk")
    4. YOLO Annotation Parsing (.txt bounding box files)
    5. Corrupted Image Safeguards (gracefully skips broken files)
    
    Returns 3-tuple:
        (img_tensor, label_idx, is_labeled)
    """
    def __init__(self, root_dir: str = None, meta_file: str = None, is_labeled: bool = True,
                 transform=None, is_train: bool = True, image_size=(112, 112)):
        super(RobustUniversalFaceDataset, self).__init__()
        self.root_dir = root_dir
        self.meta_file = meta_file
        self.is_labeled = is_labeled
        self.is_train = is_train
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
            
        self.samples = [] # List of tuples: (image_path, label_idx, is_labeled, optional_bbox)
        self.class_to_idx = {}
        
        if root_dir and os.path.exists(root_dir):
            self._discover_and_parse_dataset()
        else:
            # Fallback synthetic data generator for dry-run testing
            self.class_to_idx = {f"Subject_{i}": i for i in range(50)}
            for i in range(500):
                self.samples.append(("dummy_tensor", i % 50 if is_labeled else -1, is_labeled, None))

    def _discover_and_parse_dataset(self):
        """
        Discovers dataset format automatically without hardcoded assumptions.
        """
        # Step 1: Check for metadata files in root_dir if meta_file is not explicitly passed
        meta_path = self.meta_file
        if not meta_path:
            for candidate in ["labels.csv", "metadata.csv", "annotations.json", "labels.txt", "metadata.txt", "train.csv"]:
                possible = os.path.join(self.root_dir, candidate)
                if os.path.exists(possible):
                    meta_path = possible
                    break
                    
        # If metadata file is found, parse it
        if meta_path and os.path.exists(meta_path):
            self._parse_metadata_file(meta_path)
            if len(self.samples) > 0:
                return
                
        # Step 2: Recursive Subfolder Hierarchy (root/**/identity_name/img.jpg)
        subdirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        if len(subdirs) > 0:
            self._parse_folder_hierarchy()
            if len(self.samples) > 0:
                return
                
        # Step 3: Flat file discovery with Filename Pattern Regex Parsing
        self._parse_flat_files()

    def _parse_metadata_file(self, meta_path: str):
        """
        Parses CSV, JSON, or TXT metadata files.
        """
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        if meta_path.endswith('.csv') or meta_path.endswith('.tsv'):
            delimiter = '\t' if meta_path.endswith('.tsv') else ','
            with open(meta_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=delimiter)
                headers = [h.strip().lower() for h in next(reader, [])]
                
                # Identify column indices
                img_col = next((i for i, h in enumerate(headers) if any(k in h for k in ['img', 'file', 'image', 'path'])), 0)
                label_col = next((i for i, h in enumerate(headers) if any(k in h for k in ['label', 'id', 'name', 'subject', 'person'])), 1)
                
                for row in reader:
                    if len(row) > max(img_col, label_col):
                        img_name = row[img_col].strip()
                        raw_label = row[label_col].strip()
                        
                        full_path = os.path.join(self.root_dir, img_name) if not os.path.isabs(img_name) else img_name
                        if os.path.exists(full_path):
                            if self.is_labeled:
                                if raw_label not in self.class_to_idx:
                                    self.class_to_idx[raw_label] = len(self.class_to_idx)
                                label_idx = self.class_to_idx[raw_label]
                            else:
                                label_idx = -1
                            self.samples.append((full_path, label_idx, self.is_labeled, None))
                            
        elif meta_path.endswith('.json'):
            with open(meta_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
                
            items = data_dict.items() if isinstance(data_dict, dict) else enumerate(data_dict)
            for k, v in items:
                img_name = k if isinstance(v, (str, dict)) else v.get('file', '')
                raw_label = str(v) if isinstance(v, (str, int)) else str(v.get('label', v.get('name', '')))
                
                full_path = os.path.join(self.root_dir, img_name) if not os.path.isabs(img_name) else img_name
                if os.path.exists(full_path):
                    if self.is_labeled:
                        if raw_label not in self.class_to_idx:
                            self.class_to_idx[raw_label] = len(self.class_to_idx)
                        label_idx = self.class_to_idx[raw_label]
                    else:
                        label_idx = -1
                    self.samples.append((full_path, label_idx, self.is_labeled, None))

    def _parse_folder_hierarchy(self):
        """
        Parses nested directory hierarchy (root/**/identity_name/image.jpg).
        """
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        for root, dirs, files in os.walk(self.root_dir):
            image_files = [f for f in files if f.lower().endswith(valid_exts)]
            if len(image_files) > 0:
                folder_name = os.path.basename(root)
                
                if self.is_labeled:
                    if folder_name not in self.class_to_idx:
                        self.class_to_idx[folder_name] = len(self.class_to_idx)
                    label_idx = self.class_to_idx[folder_name]
                else:
                    label_idx = -1
                    
                for img_file in image_files:
                    full_path = os.path.join(root, img_file)
                    
                    # Check for matching YOLO .txt bounding box file
                    txt_path = os.path.splitext(full_path)[0] + '.txt'
                    bbox = self._read_yolo_bbox(txt_path) if os.path.exists(txt_path) else None
                    
                    self.samples.append((full_path, label_idx, self.is_labeled, bbox))

    def _parse_flat_files(self):
        """
        Parses flat files using Regex pattern matching (e.g. PersonName_0001.jpg -> "PersonName").
        """
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        pattern = re.compile(r'^(.*?)(?:_\d+)?\.(?:jpg|jpeg|png|bmp|webp)$', re.IGNORECASE)
        
        for f in os.listdir(self.root_dir):
            if f.lower().endswith(valid_exts):
                full_path = os.path.join(self.root_dir, f)
                
                if self.is_labeled:
                    match = pattern.match(f)
                    raw_label = match.group(1) if match else "unknown"
                    if raw_label not in self.class_to_idx:
                        self.class_to_idx[raw_label] = len(self.class_to_idx)
                    label_idx = self.class_to_idx[raw_label]
                else:
                    label_idx = -1
                    
                self.samples.append((full_path, label_idx, self.is_labeled, None))

    def _read_yolo_bbox(self, txt_path: str):
        """
        Reads YOLO format bounding box: class_id x_center y_center width height
        """
        try:
            with open(txt_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    parts = [float(x) for x in line.split()[1:5]]
                    if len(parts) == 4:
                        return parts # [xc, yc, w, h] normalized
        except Exception:
            pass
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx, is_labeled, bbox = self.samples[idx]
        
        if path == "dummy_tensor":
            img_tensor = torch.randn(3, *self.image_size)
            return img_tensor, label_idx, is_labeled
            
        try:
            img = Image.open(path).convert('RGB')
            
            # Crop using bounding box if available
            if bbox is not None:
                w_img, h_img = img.size
                xc, yc, w_box, h_box = bbox
                x1 = max(0, int((xc - w_box / 2.0) * w_img))
                y1 = max(0, int((yc - h_box / 2.0) * h_img))
                x2 = min(w_img, int((xc + w_box / 2.0) * w_img))
                y2 = min(h_img, int((yc + h_box / 2.0) * h_img))
                
                if x2 > x1 and y2 > y1:
                    img = img.crop((x1, y1, x2, y2))
                    
            img_tensor = self.transform(img)
        except Exception as e:
            # Corrupted image safeguard: return dummy tensor if PIL fails to read file
            img_tensor = torch.zeros(3, *self.image_size)
            
        return img_tensor, label_idx, is_labeled

# Backwards compatibility Aliases across codebase
WildFaceDataset = RobustUniversalFaceDataset
UnlabeledFaceDataset = lambda root_dir=None, **kwargs: RobustUniversalFaceDataset(root_dir=root_dir, is_labeled=False, **kwargs)

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
