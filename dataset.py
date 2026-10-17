import os
import re
import glob
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import torch
import torch.utils.data as data

try:
    import torchvision.transforms as transforms
except Exception:
    transforms = None

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


class DatasetConfigurationError(ValueError):
    """Raised when a production dataset cannot provide valid samples."""


def _require_dataset_root(root_dir: str, name: str) -> str:
    if not root_dir:
        raise FileNotFoundError(
            f"{name} is not configured. Pass an explicit dataset directory; "
            "synthetic fixtures are disabled in production."
        )
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"{name} does not exist or is not a directory: {root_dir}")
    return root_dir


def canonical_identity_name(name: str) -> str:
    """Normalize dataset condition prefixes without making conditions identities."""
    normalized = name.strip()
    normalized = re.sub(r'^(ROF_|WILD_|MASK_)', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r'(?:_wearing_(?:mask|sunglasses|hat)|_(?:mask|sunglasses|hat)|'
        r'_(?:with_mask|masked))$',
        '', normalized, flags=re.IGNORECASE,
    )
    return normalized or name.strip()

class PurePyTorchImageTransform:
    """
    Fallback Image Transform using pure PyTorch & PIL.
    Requires ZERO external C-libraries.
    """
    def __init__(self, image_size=(112, 112), is_train=True):
        self.image_size = image_size
        self.is_train = is_train

    def __call__(self, img: Image.Image) -> torch.Tensor:
        height, width = self.image_size
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
            
        if self.is_train and torch.rand(1).item() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
        img_bytes = bytearray(img.tobytes())
        tensor = torch.frombuffer(img_bytes, dtype=torch.uint8)
        tensor = tensor.view(height, width, 3).permute(2, 0, 1).float() / 255.0
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
    def __init__(self, root_dir: str = None, unlabeled_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112), allow_synthetic: bool = False):
        super(UnlabeledWildDataset, self).__init__()
        self.unlabeled_dir = root_dir if root_dir is not None else unlabeled_dir
        self.image_size = image_size
        self.allow_synthetic = allow_synthetic
        self.transform = transform if transform is not None else get_default_transform(image_size, is_train)
        self.samples = []
        self.rescan()

    def rescan(self):
        """
        Dynamically rescans the unlabeled directory to index newly arrived images uploaded in parallel.
        Excludes bounding-box datasets (e.g., masked-face-detection-wider-dataset) handled by BoundingBoxFaceDataset.
        """
        prev_count = len(self.samples)
        self.samples = []
        if self.unlabeled_dir and os.path.exists(self.unlabeled_dir):
            for root, _, files in os.walk(self.unlabeled_dir):
                for f in sorted(files):
                    if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                        self.samples.append(os.path.join(root, f))
                        
        if len(self.samples) == 0 and self.allow_synthetic:
            for _ in range(500):
                self.samples.append("dummy_unlabeled")
        elif len(self.samples) == 0:
            raise DatasetConfigurationError(
                f"No valid images were found in unlabeled dataset: {self.unlabeled_dir}"
            )
        if len(self.samples) != prev_count:
            print(f"[Dynamic Unlabeled Rescan] Indexed {len(self.samples)} images (New: +{len(self.samples) - prev_count})")
        return len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Unlabeled dataset index out of range: {idx}")
        path = self.samples[idx]
        if path == "dummy_unlabeled":
            img_tensor = torch.randn(3, *self.image_size)
        else:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
            except Exception as exc:
                raise RuntimeError(f"Could not decode unlabeled image {path}: {exc}") from exc
                
        return img_tensor, -1, False


class NameLabeledFaceDataset(data.Dataset):
    """
    Dynamic Dataset Loader for 'name_label/' directory.
    Supports dynamic rescanning of constantly growing labeled datasets uploaded in parallel.
    """
    def __init__(self, root_dir: str = None, name_label_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112), allow_synthetic: bool = False):
        super(NameLabeledFaceDataset, self).__init__()
        self.name_label_dir = root_dir if root_dir is not None else name_label_dir
        self.image_size = image_size
        self.allow_synthetic = allow_synthetic
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
        self.class_to_idx = {}
        if self.name_label_dir and os.path.exists(self.name_label_dir):
            self._parse_name_label_directory()
        elif self.allow_synthetic:
            self.class_to_idx = {f"Subject_{i}": i for i in range(50)}
            for i in range(500):
                self.samples.append(("dummy_labeled", i % 50))
        else:
            raise FileNotFoundError(
                f"Labeled dataset does not exist or is not configured: {self.name_label_dir}"
            )

        if not self.samples:
            raise DatasetConfigurationError(
                f"No valid labelled images were found in: {self.name_label_dir}"
            )
                
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
            dirs.sort()
            image_files = sorted(f for f in files if f.lower().endswith(VALID_IMAGE_EXTENSIONS))
            if len(image_files) > 0:
                folder_name = canonical_identity_name(prefix + os.path.basename(root))
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
                    identity_name = canonical_identity_name("MASK_" + raw_name)
                    
                    if identity_name not in self.class_to_idx:
                        self.class_to_idx[identity_name] = len(self.class_to_idx)
                    label_idx = self.class_to_idx[identity_name]
                    self.samples.append((os.path.join(root, f), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Labeled dataset index out of range: {idx}")
        path, label_idx = self.samples[idx]
        if path == "dummy_labeled":
            img_tensor = torch.randn(3, *self.image_size)
        else:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
            except Exception as exc:
                raise RuntimeError(f"Could not decode labelled image {path}: {exc}") from exc
                
        return img_tensor, label_idx, True


class BoundingBoxFaceDataset(data.Dataset):
    """
    Dynamic Dataset Loader for 'bb_label/' directory.
    Supports dynamic rescanning of bounding box datasets uploaded in parallel.
    """
    def __init__(self, root_dir: str = None, bb_label_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112), allow_synthetic: bool = False):
        super(BoundingBoxFaceDataset, self).__init__()
        self.bb_label_dir = root_dir if root_dir is not None else bb_label_dir
        self.image_size = image_size
        self.allow_synthetic = allow_synthetic
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
        elif self.allow_synthetic:
            for i in range(200):
                self.samples.append(("dummy_bb", None, 0))
        else:
            raise FileNotFoundError(
                f"Bounding-box dataset does not exist or is not configured: {self.bb_label_dir}"
            )

        if not self.samples:
            raise DatasetConfigurationError(
                f"No valid bounding-box images were found in: {self.bb_label_dir}"
            )
                
        if len(self.samples) != prev_count:
            print(f"[Dynamic BoundingBox Rescan] Indexed {len(self.samples)} bounding box images (New: +{len(self.samples) - prev_count})")
        return len(self.samples)

    def _parse_bb_label_directory(self):
        search_dirs = []
        if self.bb_label_dir and os.path.exists(self.bb_label_dir):
            search_dirs.append(self.bb_label_dir)
            parent_dir = os.path.dirname(self.bb_label_dir.rstrip('/'))
            wider_cand = os.path.join(parent_dir, "unlabeled", "masked-face-detection-wider-dataset")
            if os.path.exists(wider_cand) and wider_cand not in search_dirs:
                search_dirs.append(wider_cand)

        for sdir in search_dirs:
            if sdir and os.path.exists(sdir):
                for root, _, files in os.walk(sdir):
                    for f in sorted(files):
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
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Bounding-box dataset index out of range: {idx}")
        path, bboxes, class_id = self.samples[idx]
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
        except Exception as exc:
            raise RuntimeError(f"Could not decode bounding-box image {path}: {exc}") from exc
            
        return img_tensor, class_id, True


class UnifiedWildFaceDataset(data.Dataset):
    """
    Unified Dynamic Dataset Loader combining all datastreams with automatic epoch-wise rescanning.
    """
    def __init__(self, root_dir: str = None, meta_file: str = None, is_labeled: bool = True, transform=None, is_train: bool = True, image_size=(112, 112), allow_synthetic: bool = False):
        super(UnifiedWildFaceDataset, self).__init__()
        self.root_dir = _require_dataset_root(root_dir, "unified face dataset")
        self.is_labeled = is_labeled
        self.is_train = is_train
        if is_labeled:
            labelled_root = os.path.join(root_dir, "name_label")
            labelled_root = labelled_root if os.path.isdir(labelled_root) else root_dir
            self.primary_ds = NameLabeledFaceDataset(
                name_label_dir=labelled_root,
                transform=transform, is_train=is_train, image_size=image_size,
                allow_synthetic=allow_synthetic,
            )
            self.class_to_idx = self.primary_ds.class_to_idx
        else:
            unlabeled_root = os.path.join(root_dir, "unlabeled")
            unlabeled_root = unlabeled_root if os.path.isdir(unlabeled_root) else root_dir
            self.primary_ds = UnlabeledWildDataset(
                unlabeled_dir=unlabeled_root,
                transform=transform, is_train=is_train, image_size=image_size,
                allow_synthetic=allow_synthetic,
            )
            self.class_to_idx = {}

    def rescan(self):
        """
        Rescans all sub-datasets to pick up newly arrived parallel file uploads.
        """
        count = self.primary_ds.rescan()
        self.class_to_idx = getattr(self.primary_ds, "class_to_idx", {})
        return count

    def __len__(self):
        return len(self.primary_ds)

    def __getitem__(self, idx):
        return self.primary_ds[idx]


RobustUniversalFaceDataset = UnifiedWildFaceDataset
WildFaceDataset = NameLabeledFaceDataset
UnlabeledFaceDataset = UnlabeledWildDataset

class Phase2UnifiedAttributeDataset(data.Dataset):
    """CelebA-only attribute dataset with real labels and no fake wild labels.

    Attribute supervision is a separate task from identity recognition. Wild
    images without annotation are intentionally excluded instead of receiving
    random targets. Occlusion masks are not fabricated here; a future mask
    dataset can be added as a separate task with its own annotation contract.
    """

    def __init__(self, root_dir: str = None, celeba_dir: str = None, attr_file: str = None,
                 is_train: bool = True, image_size=(112, 112)):
        super().__init__()
        self.root_dir = _require_dataset_root(root_dir or celeba_dir, "CelebA attribute dataset")
        self.image_size = image_size
        self.transform = get_default_transform(image_size, is_train)
        self.samples = []

        parent_dir = os.path.dirname(self.root_dir.rstrip('/'))
        if attr_file is None:
            candidates = [
                os.path.join(self.root_dir, "list_attr_celeba.csv"),
                os.path.join(self.root_dir, "list_attr_celeba.txt"),
                os.path.join(self.root_dir, "Anno", "list_attr_celeba.txt"),
                os.path.join(parent_dir, "list_attr_celeba.csv"),
                os.path.join(parent_dir, "list_attr_celeba.txt"),
            ]
            attr_file = next((path for path in candidates if os.path.isfile(path)), None)
        if not attr_file or not os.path.isfile(attr_file):
            raise FileNotFoundError(
                "CelebA attribute annotations were not found. Expected list_attr_celeba.csv or .txt."
            )

        image_roots = [
            self.root_dir,
            os.path.join(self.root_dir, "img_align_celeba"),
            os.path.join(self.root_dir, "img_align_celeba", "img_align_celeba"),
        ]
        raw_entries = []
        if attr_file.lower().endswith('.csv'):
            import csv
            with open(attr_file, 'r', encoding='utf-8', errors='strict') as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for row in reader:
                    if len(row) < 41:
                        continue
                    raw_entries.append((row[0], torch.tensor([1.0 if int(x) == 1 else 0.0 for x in row[1:41]], dtype=torch.float32)))
        else:
            with open(attr_file, 'r', encoding='utf-8', errors='strict') as handle:
                lines = handle.readlines()
            for line in lines[2:]:
                parts = line.strip().split()
                if len(parts) >= 41:
                    raw_entries.append((parts[0], torch.tensor([1.0 if int(x) == 1 else 0.0 for x in parts[1:41]], dtype=torch.float32)))

        for image_name, target in raw_entries:
            image_path = next((os.path.join(root, image_name) for root in image_roots if os.path.isfile(os.path.join(root, image_name))), None)
            if image_path:
                self.samples.append((image_path, target))
        if not self.samples:
            raise DatasetConfigurationError(f"No CelebA images matched annotations in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"CelebA dataset index out of range: {idx}")
        path, target = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            return self.transform(image), target
        except Exception as exc:
            raise RuntimeError(f"Could not decode CelebA image {path}: {exc}") from exc


CelebAAttributeDataset = Phase2UnifiedAttributeDataset
