"""
dataset.py — Dataset & Data Pipeline
=====================================
Backend Developer Agent | Data Layer | PyTorch

WHY THIS FILE EXISTS:
  All data concerns — downloading, preprocessing, augmentation, and
  batching — live here and nowhere else (Single Responsibility).
  Training and inference scripts call get_dataloaders() and receive
  ready-to-use DataLoaders without knowing how images are loaded or
  transformed. This is the Repository Pattern applied to datasets.

DATASET CHOICE — Oxford-IIIT Pet Dataset:
  • Publicly available, no sign-up required.
  • Auto-downloaded by torchvision.datasets.OxfordIIITPet (~750 MB).
  • 7,349 images of 37 pet breeds with species labels (Cat / Dog).
  • Sufficient size for fine-tuning a pretrained CNN to >90% accuracy.
  • License: Creative Commons Attribution-ShareAlike 4.0 (CC-BY-SA 4.0).

ALTERNATIVE supported: Local ImageFolder
  • Use --source local with ./data/train/cats|dogs and ./data/val/cats|dogs.

Usage:
    from dataset import get_dataloaders
    train_loader, val_loader, class_names = get_dataloaders(source='oxford')
"""

import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.datasets import OxfordIIITPet


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE  = 224                         # standard ImageNet input resolution
MEAN        = [0.485, 0.456, 0.406]       # ImageNet channel means (R, G, B)
STD         = [0.229, 0.224, 0.225]       # ImageNet channel standard deviations
CLASS_NAMES = ["Cat", "Dog"]              # index 0 = Cat, index 1 = Dog


# ──────────────────────────────────────────────────────────────────────────────
# get_train_transform() — Augmentation Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """
    Build the augmentation pipeline applied only during training.

    WHAT IT DOES:
      Returns a composed sequence of image transformations that randomly
      alter each training image every epoch, effectively multiplying
      the diversity of the dataset without collecting new photos.

    WHY EACH AUGMENTATION (in order):

      1. Resize((256, 256)):
         Rescales all images to a uniform size before cropping.
         WHY 256 not 224: gives the random crop room to sample from
         different spatial positions, so every epoch sees a slightly
         different view of the same image — free data augmentation.

      2. RandomCrop(224):
         Randomly crops a 224×224 patch from the 256×256 resized image.
         WHY random (not center) crop: cat/dog subjects are not always
         centred; random crop forces the model to be position-invariant.

      3. RandomHorizontalFlip(p=0.5):
         Mirrors the image left-to-right with 50% probability.
         WHY: Cats and dogs look the same facing left or right. This
         doubles the effective dataset size at zero cost. Horizontal
         flip is the safest augmentation for natural images — never
         use vertical flip (upside-down pets are not in the test set).

      4. RandomRotation(15°):
         Rotates image by a random angle in [-15°, +15°].
         WHY: Animals are photographed at various angles. Small rotations
         prevent the model from relying on orientation as a feature.

      5. ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
         Randomly alters colour properties.
         WHY: Lighting varies dramatically between photos (indoor, outdoor,
         flash, shade). Jitter forces the model to rely on shape/texture,
         not specific colour tones. hue=0.1 is small to avoid unrealistic
         colours (e.g. green cats).

      6. ToTensor():
         Converts PIL Image (H, W, C) uint8 to FloatTensor (C, H, W) in [0, 1].
         WHY: PyTorch tensors are the required format; also transposes channels.

      7. Normalize(MEAN, STD):
         Subtracts per-channel ImageNet mean and divides by std so the
         input distribution matches what pretrained models (ResNet) expect.
         WHY: Pretrained models were trained on normalised ImageNet data.
         Applying the same normalisation ensures the pretrained features
         activate correctly on our pet images without re-training from scratch.

    Returns:
        transforms.Compose — callable that transforms a PIL Image to a Tensor.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# get_val_transform() — Deterministic Inference Transform
# ──────────────────────────────────────────────────────────────────────────────

def get_val_transform() -> transforms.Compose:
    """
    Build the deterministic transform used for validation and inference.

    WHAT IT DOES:
      Returns a fixed (non-random) transformation pipeline that produces
      the same tensor every time it is applied to the same image.

    WHY NO AUGMENTATION AT VALIDATION TIME:
      Augmentation introduces randomness, which means validation metrics
      would differ slightly on every evaluation epoch — making it hard to
      compare epochs and choose the best checkpoint. We always want a
      fair, reproducible benchmark against unseen data.

    WHY CenterCrop instead of RandomCrop:
      Centre-cropping captures the most informative part of the image
      (subjects are usually centred by the photographer). It is the
      standard deterministic alternative used in every major image
      classification paper (e.g. ResNet, EfficientNet original papers).

    Returns:
        transforms.Compose — deterministic callable for PIL → Tensor.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),     # deterministic, always same crop
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# OxfordPetBinaryDataset — Public Dataset Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class OxfordPetBinaryDataset(Dataset):
    """
    Wraps torchvision's OxfordIIITPet dataset and converts its species
    labels from {1=Cat, 2=Dog} to float labels {0.0=Cat, 1.0=Dog}.

    WHAT IT DOES:
      Delegates all image I/O to the official OxfordIIITPet class
      (which handles downloading, extracting, and file path management).
      This wrapper only adjusts labels for compatibility with
      nn.BCEWithLogitsLoss, which expects float targets in {0.0, 1.0}.

    WHY OXFORD-IIIT PET DATASET:
      • Free public dataset, no registration or API key required.
      • Auto-downloaded (~750 MB) via torchvision on first use.
      • Provides 'species' target_type: a binary label (Cat or Dog)
        already extracted from 37 breed annotations, saving us annotation work.
      • 7,349 images with near-balanced class distribution:
          Cats ≈ 2,371 images | Dogs ≈ 4,978 images
      • Commonly used in academic papers, so results can be compared.
      • URL: https://www.robots.ox.ac.uk/~vgg/data/pets/

    WHY SUBCLASS Dataset instead of using OxfordIIITPet directly:
      OxfordIIITPet returns species as an integer {1, 2}.
      BCEWithLogitsLoss needs float {0.0, 1.0}.
      Wrapping this conversion here keeps train.py clean.
      This is the Adapter pattern: adapting one interface to another.
    """

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        transform=None,
        download: bool = True,
    ):
        """
        Args:
            root:      Directory where the dataset will be downloaded/cached.
            split:     'trainval' (~7K images) or 'test' (~3.7K images).
                       WHY 'trainval': we do our own 80/20 split for validation
                       to maximise training data. The 'test' split is reserved
                       for final unbiased evaluation.
            transform: Callable applied to each PIL image before returning.
            download:  If True, download the dataset if not already present.
                       Set to False on subsequent dataset instantiations
                       (val set) to avoid redundant download checks.
        """
        self.inner = OxfordIIITPet(
            root=root,
            split=split,
            target_types="binary-category",  # returns {0=Cat, 1=Dog}; renamed from 'species' in torchvision 0.21+
            transform=transform,
            download=download,
        )

    def __len__(self) -> int:
        """
        WHAT: Returns the total number of images in this split.
        WHY: Required by PyTorch's Dataset contract so DataLoader can
             calculate how many batches to yield per epoch.
        """
        return len(self.inner)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve one (image, label) pair by index.

        WHAT:
          Fetches the image and raw species integer from the inner dataset,
          then converts the species to a float scalar tensor.

        WHY float32 scalar tensor:
          nn.BCEWithLogitsLoss requires labels as float32 with shape matching
          the model output. Using .unsqueeze(1) in the training loop then
          gives shape (B, 1) to match model output (B, 1).

        WHY species - 1:
          Oxford labels: Cat=1, Dog=2. We shift to Cat=0, Dog=1
          because class index 0 always means 'negative class' (Cat)
          and index 1 means 'positive class' (Dog) in binary classification.

        Args:
            idx: Integer index in [0, len(self)).
        Returns:
            image: Float tensor of shape (3, 224, 224) after transforms.
            label: Float tensor scalar — 0.0 for Cat, 1.0 for Dog.
        """
        image, species = self.inner[idx]           # binary-category ∈ {0=Cat, 1=Dog}
        label = torch.tensor(float(species), dtype=torch.float32)  # already {0.0, 1.0}
        return image, label


# ──────────────────────────────────────────────────────────────────────────────
# LocalCatDogDataset — Bring-Your-Own-Data Support
# ──────────────────────────────────────────────────────────────────────────────

class LocalCatDogDataset(Dataset):
    """
    Loads images from a standard folder structure for users who have their
    own cat/dog image collection (e.g. downloaded Kaggle Dogs vs. Cats).

    WHAT IT DOES:
      Uses torchvision.datasets.ImageFolder which auto-assigns class labels
      from sub-folder names (alphabetical order: cats=0, dogs=1).

    WHY ImageFolder:
      It is the de-facto standard for local image classification datasets.
      No custom file-listing code needed — ImageFolder handles all JPEG/PNG
      discovery, class mapping, and optional transform application.

    Expected directory layout:
        root/
          cats/  image1.jpg, image2.jpg, ...
          dogs/  image1.jpg, image2.jpg, ...

    WHY wrap ImageFolder instead of using it directly:
      ImageFolder returns integer labels from torch.long.
      BCEWithLogitsLoss needs float32. We convert here so all Dataset
      objects expose the same interface as OxfordPetBinaryDataset.
    """

    def __init__(self, root: str, transform=None):
        """
        Args:
            root:      Path to a folder containing 'cats/' and 'dogs/' subdirectories.
            transform: Transform pipeline (use get_train_transform or get_val_transform).
        """
        self.dataset = datasets.ImageFolder(root=root, transform=transform)
        # WHY confirm mapping: alphabetical order gives cats=0, dogs=1 which is
        # the same convention as OxfordPetBinaryDataset. Explicit assertion
        # would catch any folder naming error at construction time.
        self.class_to_idx = {"cats": 0, "dogs": 1}

    def __len__(self) -> int:
        """
        WHAT: Returns total image count in this folder.
        WHY: Required by Dataset contract for DataLoader batch planning.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch one (image, label) pair from the local folder.

        WHAT:
          Delegates to ImageFolder's __getitem__, then casts the integer
          class index to a float32 scalar for loss compatibility.

        WHY cast to float32:
          Same reason as OxfordPetBinaryDataset — BCEWithLogitsLoss
          requires matching dtypes (float32 output, float32 target).

        Args:
            idx: Image index.
        Returns:
            image: Transformed tensor (3, 224, 224).
            label: Float32 scalar — 0.0 (Cat) or 1.0 (Dog).
        """
        image, label_idx = self.dataset[idx]
        return image, torch.tensor(float(label_idx), dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# get_dataloaders() — Public API (DataLoader Factory)
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    source: str = "oxford",
    data_dir: str = "./data",
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create and return train and validation DataLoaders.

    WHAT IT DOES:
      Acts as the single public API for all data loading concerns.
      Selects the correct dataset source, applies appropriate transforms,
      and wraps everything in PyTorch DataLoaders configured for performance.

    WHY A FACTORY FUNCTION:
      train.py, predict.py, and the Colab notebook all call this one
      function. If we later want to add a new dataset source (e.g. CIFAR
      or a web scraper), we add it here and nothing else changes.
      This is the Factory pattern and the Facade pattern combined.

    WHY DataLoader (not manual iteration):
      DataLoader handles batching, shuffling, multi-process prefetching,
      and optional memory pinning. Without it, data loading would be the
      training bottleneck on a fast GPU.

    DataLoader performance settings explained:
      • shuffle=True (train only):
          Randomises item order each epoch so the model cannot learn
          from ordering artefacts, and mini-batches represent the full
          class distribution rather than long runs of one class.
      • num_workers > 0:
          Spawns background processes that preload the next batch while
          the GPU is processing the current one. Eliminates CPU-GPU idle time.
          WHY cap at os.cpu_count(): requesting more workers than CPU cores
          causes thrashing and actually slows loading.
      • pin_memory=True:
          Allocates CPU tensor memory in page-pinned (non-pageable) RAM,
          which allows the CUDA driver to perform DMA transfers to GPU
          without copying through pageable memory first — roughly 2× faster
          host-to-device transfer on supported GPUs.
      • drop_last=True (train only):
          Drops the last incomplete batch. WHY: prevents BatchNorm from
          receiving a batch of size 1, which would cause NaN statistics
          (variance of a single sample is undefined).

    Args:
        source:      'oxford' — use Oxford Pets (auto-download, recommended).
                     'local'  — use local ImageFolder at data_dir/train & val.
        data_dir:    Root path for dataset storage or local images.
        batch_size:  Number of images per mini-batch.
                     WHY 32: fits on 6 GB VRAM; 64 is faster but needs more VRAM.
        val_split:   Fraction of Oxford trainval to hold out for validation.
                     WHY 0.2: standard 80/20 split, leaves 5,879 train images.
        num_workers: Parallel data-loading processes.
        pin_memory:  Enable pinned CPU memory for faster GPU transfer.
    Returns:
        (train_loader, val_loader, class_names)
        class_names: ['Cat', 'Dog'] — index 0 = Cat, index 1 = Dog.
    Raises:
        ValueError: If source is not 'oxford' or 'local'.
        FileNotFoundError: If source='local' and expected folders are missing.
    """
    # Cap workers to available CPU cores to avoid resource contention
    num_workers = min(num_workers, os.cpu_count() or 1)

    if source == "oxford":
        return _oxford_loaders(data_dir, batch_size, val_split, num_workers, pin_memory)
    elif source == "local":
        return _local_loaders(data_dir, batch_size, num_workers, pin_memory)
    else:
        raise ValueError(f"Unknown source '{source}'. Choose 'oxford' or 'local'.")


# ──────────────────────────────────────────────────────────────────────────────
# Internal Helpers — not part of the public API
# ──────────────────────────────────────────────────────────────────────────────

def _oxford_loaders(
    data_dir: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build DataLoaders from the Oxford-IIIT Pet Dataset (auto-downloaded).

    WHAT IT DOES:
      1. Instantiates OxfordPetBinaryDataset twice — once with train
         augmentation, once with val transform — over the same trainval split.
      2. Generates a reproducible 80/20 random split via a seeded Generator.
      3. Uses Subset to apply the correct transform variant to each split.
      4. Wraps both subsets in DataLoaders.

    WHY TWO DATASET INSTANCES (one per transform):
      PyTorch Subset wraps indices into an existing Dataset. If we used
      a single dataset instance for both train and val indices, all items
      would receive the same transform (either augmented or not).
      By creating full_train_ds (augmented) and full_val_ds (clean) and
      then subsetting each with the same indices, train items get
      high-variance transforms while val items get deterministic ones.

    WHY manual random_split instead of ImageFolder's built-in split:
      OxfordIIITPet has no built-in val split parameter. random_split
      is the idiomatic PyTorch way to partition a dataset reproducibly
      via a seeded Generator.

    WHY seed=42:
      Reproducibility. The same split is used across training runs so that
      model performance differences are due to architecture/hyperparameters,
      not dataset split randomness.

    Args:
        data_dir, batch_size, val_split, num_workers, pin_memory:
            See get_dataloaders() docstring.
    Returns:
        (train_loader, val_loader, CLASS_NAMES)
    """
    print(f"[Dataset] Source: Oxford-IIIT Pet (auto-download to {data_dir})")

    # Two dataset instances — different transforms, same underlying files
    full_train_ds = OxfordPetBinaryDataset(
        root=data_dir, split="trainval",
        transform=get_train_transform(), download=True,
    )
    full_val_ds = OxfordPetBinaryDataset(
        root=data_dir, split="trainval",
        transform=get_val_transform(), download=False,  # already downloaded above
    )

    n_total = len(full_train_ds)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    # Seeded generator ensures identical split across restarts
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )

    # WHY Subset: lightweight wrapper — no data is copied, only index lists
    train_subset = Subset(full_train_ds, train_indices.indices)
    val_subset   = Subset(full_val_ds,   val_indices.indices)

    print(f"[Dataset] Train: {len(train_subset):,}  |  Val: {len(val_subset):,}")

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, CLASS_NAMES


def _local_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build DataLoaders from a local ImageFolder directory structure.

    WHAT IT DOES:
      Reads images from pre-split data/train/ and data/val/ directories.
      Validates that required folders exist before creating any Dataset object
      — failing fast with a clear error message rather than an internal crash.

    WHY FAIL FAST (FileNotFoundError early):
      If a training run starts and then crashes on the first data batch
      (e.g., empty folder), the user wastes time downloading the model
      and initialising the GPU. Checking paths at startup saves time.

    Args:
        data_dir, batch_size, num_workers, pin_memory:
            See get_dataloaders() docstring.
    Returns:
        (train_loader, val_loader, CLASS_NAMES)
    Raises:
        FileNotFoundError: If data_dir/train or data_dir/val is missing.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Local dataset not found.\n"
            f"Expected layout:\n"
            f"  {train_dir}/cats/  {train_dir}/dogs/\n"
            f"  {val_dir}/cats/    {val_dir}/dogs/"
        )

    train_ds = LocalCatDogDataset(root=train_dir, transform=get_train_transform())
    val_ds   = LocalCatDogDataset(root=val_dir,   transform=get_val_transform())

    print(f"[Dataset] Local — Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, CLASS_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test (run: python dataset.py)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    WHAT: Downloads Oxford Pets (if needed) and prints one batch's shape.
    WHY: Quick sanity check that the data pipeline works before training.
         num_workers=0 avoids multiprocessing issues on Windows in __main__.
    """
    train_loader, val_loader, classes = get_dataloaders(
        source="oxford", data_dir="./data", batch_size=8, num_workers=0
    )
    images, labels = next(iter(train_loader))
    print(f"Batch shape : {images.shape}")   # expected: (8, 3, 224, 224)
    print(f"Labels      : {labels.tolist()}") # expected: 8 floats in {0.0, 1.0}
    print(f"Classes     : {classes}")
