"""
model.py — Neural Network Model Definitions
===========================================
Backend Developer Agent | SOLID Architecture | PyTorch

WHY THIS FILE EXISTS:
  Model definitions are isolated here (Single Responsibility Principle).
  Training, data loading, and inference scripts import models via
  the build_model() factory — they never instantiate classes directly.
  This means you can swap the architecture in one place without touching
  any other part of the pipeline.

Two model options are provided:
  1. CatDogCNN       — Custom lightweight CNN (educational, fast, ~6M params)
  2. get_resnet18()  — Transfer learning with ResNet-18 (~11M params, recommended)
  3. get_resnet50()  — Transfer learning with ResNet-50 (~25M params, highest accuracy)

Usage:
    from model import build_model
    model = build_model('resnet18')   # recommended default
    model = build_model('custom_cnn') # educational / low-resource
    model = build_model('resnet50')   # maximum accuracy
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Literal


# ──────────────────────────────────────────────────────────────────────────────
# ConvBlock — Reusable Building Block
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    A single convolutional block: Conv2D → BatchNorm2D → ReLU → (optional MaxPool).

    WHAT IT DOES:
      Applies a 2D convolution to extract spatial features from the image,
      normalises the activations, applies non-linearity, then optionally
      downsamples the spatial dimensions with max pooling.

    WHY EACH COMPONENT:
      • Conv2d (kernel=3, pad=1):
          3×3 kernels are the gold standard in modern CNNs (VGG, ResNet).
          padding=1 keeps the spatial size the same before pooling.
          They are small enough to train quickly yet expressive enough to
          capture edges, textures, and object parts.

      • BatchNorm2d:
          Normalises each mini-batch's activations so the network trains
          faster and generalises better. It reduces the need for careful
          weight initialisation and acts as a mild regulariser.
          WHY chosen: it is now standard in all production CNNs.

      • ReLU (inplace=True):
          Introduces non-linearity, enabling the network to learn
          arbitrary complex mappings. inplace=True saves memory by
          modifying tensors in place instead of allocating new ones.
          WHY ReLU over sigmoid/tanh: avoids vanishing gradient problem.

      • MaxPool2d (2×2, stride 2):
          Halves the spatial dimensions (height × width) at each block,
          which reduces computation and forces the network to learn
          increasingly abstract features. Optional so we can skip pooling
          on the last block if needed.

    WHY THIS CLASS EXISTS:
      Repeating Conv → BN → ReLU → Pool manually would violate DRY.
      Encapsulating it as a module makes the architecture easy to read
      and change (Open/Closed Principle).
    """

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        """
        Args:
            in_channels:  Number of input feature maps.
            out_channels: Number of output feature maps (filters to learn).
            pool:         If True, apply 2×2 MaxPool after ReLU.
        """
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Conv → BN → ReLU → (MaxPool).

        WHAT: Feeds the input tensor through every layer in self.block sequentially.
        WHY: nn.Sequential handles the chaining; no manual intermediate variables needed.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).
        Returns:
            Output tensor — (B, out_channels, H/2, W/2) if pool=True,
                            else (B, out_channels, H, W).
        """
        return self.block(x)


# ──────────────────────────────────────────────────────────────────────────────
# CatDogCNN — Custom Architecture
# ──────────────────────────────────────────────────────────────────────────────

class CatDogCNN(nn.Module):
    """
    Custom Convolutional Neural Network for binary image classification.

    WHAT IT DOES:
      Learns to classify 224×224 RGB images as Cat (0) or Dog (1)
      using 4 stacked ConvBlocks followed by a fully connected classifier.
      Outputs a single raw logit; apply torch.sigmoid() to get probability.

    WHY A CUSTOM CNN:
      • Educational value — shows every component explicitly.
      • Lighter than ResNet — useful when GPU memory is limited.
      • No pretrained weight dependency — trains from scratch on pets data.
      • 4 blocks: each doubles the channel count while halving spatial size,
        a standard progressive design that worked well in VGGNet (Simonyan 2015).

    Architecture flow  (input 224×224×3):
      ConvBlock(3   → 32)   → (B, 32,  112, 112)
      ConvBlock(32  → 64)   → (B, 64,   56,  56)
      ConvBlock(64  → 128)  → (B, 128,  28,  28)
      ConvBlock(128 → 256)  → (B, 256,  14,  14)
      AdaptiveAvgPool(4,4)  → (B, 256,   4,   4)
      Flatten               → (B, 4096)
      Linear(4096 → 512)    → (B, 512)
      Dropout + Linear      → (B, 128)
      Dropout + Linear      → (B, 1)     ← final logit

    Loss: nn.BCEWithLogitsLoss (logit in, no sigmoid needed — numerically stable).
    """

    def __init__(self, dropout_rate: float = 0.4):
        """
        Args:
            dropout_rate: Probability of zeroing a neuron during training.
                          WHY 0.4: empirically reduces overfitting without
                          slowing convergence. Halved in the second dropout
                          layer because feature count is already reduced.
        """
        super().__init__()

        # ── Feature Extractor ──────────────────────────────────────────────────
        # WHY AdaptiveAvgPool2d instead of a fixed FC after flatten:
        #   Adaptive pooling collapses any spatial size to (4,4), so the model
        #   accepts any image size >= 32×32 without code changes.
        self.features = nn.Sequential(
            ConvBlock(3,   32),    # detects low-level: edges, colour blobs
            ConvBlock(32,  64),    # detects mid-level: textures, patterns
            ConvBlock(64,  128),   # detects high-level: shapes, fur patterns
            ConvBlock(128, 256),   # detects semantic: ears, snouts, whiskers
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # ── Classifier Head ────────────────────────────────────────────────────
        # WHY two hidden layers (4096→512→128→1):
        #   A single huge layer easily overfits. Two smaller ones with dropout
        #   in between progressively compress the representation and regularise.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, 1),   # single logit → sigmoid → P(Dog)
        )

        # Apply He/Kaiming weight initialisation
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise convolutional and linear layer weights.

        WHAT: Sets Conv2d weights to Kaiming Normal, BatchNorm to identity (1,0),
              and Linear to Xavier Uniform.

        WHY Kaiming (He) init for Conv:
          ReLU activations are non-symmetric; Kaiming Normal preserves
          variance across layers specifically for ReLU — demonstrated to
          speed up convergence vs random normal (He et al., 2015).

        WHY Xavier Uniform for Linear:
          Blances variance for symmetric activations at the FC stage;
          standard choice when the distribution is not ReLU-dominated.

        WHY initialise at all:
          Poor starting weights (too large or too small) cause gradients
          to vanish or explode, especially in deeper networks. Proper init
          is essential for stable training from scratch.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)   # scale = 1 (identity at init)
                nn.init.zeros_(m.bias)    # shift = 0 (identity at init)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image tensor → raw logit.

        WHAT: Passes input through the feature extractor, then the classifier
              to produce an unbounded scalar logit per image.

        WHY return a logit (not sigmoid probability):
          BCEWithLogitsLoss fuses sigmoid + BCE in a numerically stable
          way (log-sum-exp trick). Returning raw logits is the PyTorch
          idiomatic choice for binary classification.

        Args:
            x: Input image batch, shape (B, 3, 224, 224).
        Returns:
            Raw logit tensor, shape (B, 1).
            Apply torch.sigmoid() to convert to probability.
        """
        features = self.features(x)
        return self.classifier(features)

    def count_parameters(self) -> int:
        """
        Count trainable parameters.

        WHAT: Sums the element counts of all parameter tensors that require grad.
        WHY: Useful for comparing model sizes and estimating memory requirements.

        Returns:
            Integer number of trainable scalar parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Transfer Learning Factories
# ──────────────────────────────────────────────────────────────────────────────

def get_resnet18_model(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """
    Build a ResNet-18 model adapted for binary cat/dog classification.

    WHAT IT DOES:
      Loads ResNet-18 (optionally with ImageNet pretrained weights) and
      replaces the original 1000-class FC head with a binary classifier:
        Dropout(0.3) → Linear(512 → 1)

    WHY ResNet-18 (not VGG, not EfficientNet):
      • Residual connections solve the vanishing gradient problem for
        deeper networks, letting every layer receive a strong gradient
        regardless of depth (He et al., 2016).
      • 18 layers: fast enough to iterate on a single consumer GPU
        (RTX 3060 trains in ~1–2 min/epoch on Oxford Pets).
      • Only ~11M parameters — far less prone to overfitting on the
        ~6K training images we have than a 50+ layer model.
      • ImageNet pretraining: the backbone already understands edges,
        textures, and object parts. Fine-tuning only the last layer(s)
        achieves >92% accuracy in 15 epochs — vs training from scratch
        which needs >50 epochs for comparable results.

    WHY freeze_backbone option:
      Freezing forces only the new FC head to train, which is fast
      (~10× less compute) but sacrifices accuracy if the ImageNet
      features don't transfer perfectly. Default False lets the whole
      network adapt to pet images.

    Args:
        pretrained:      Load ImageNet weights. True is strongly recommended.
        freeze_backbone: If True, only the new FC head is trainable.
    Returns:
        Modified ResNet-18 model with 1-output FC head.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        # Disable gradient tracking for all backbone parameters
        for param in model.parameters():
            param.requires_grad = False

    # WHY replace only model.fc:
    #   All convolutional backbone layers stay intact (and pretrained).
    #   Only the final FC (originally 512→1000) is swapped to (512→1),
    #   which is the standard transfer-learning recipe for image tasks.
    in_features = model.fc.in_features    # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),                # prevents co-adaptation of head neurons
        nn.Linear(in_features, 1),        # binary output logit
    )
    return model


def get_resnet50_model(pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet-50 model adapted for binary cat/dog classification.

    WHAT IT DOES:
      Same transfer-learning recipe as get_resnet18_model() but uses
      ResNet-50, which is 3× deeper (~25M params vs 11M).

    WHY ResNet-50 vs ResNet-18:
      • Higher capacity → learns finer-grained feature differences
        (e.g., fine fur textures vs whisker patterns).
      • Bottleneck residual blocks use 1×1 convolutions to reduce
        computation, making depth more efficient than sheer width.
      • Expected accuracy: ~95–97% vs ~92–95% for ResNet-18.
      • Trade-off: ~2× slower to train per epoch; may overfit on small
        datasets without strong augmentation. Use when GPU VRAM >= 8 GB.

    Args:
        pretrained: Load ImageNet V2 weights (higher quality than V1).
    Returns:
        Modified ResNet-50 model with 1-output FC head.
    """
    # WHY IMAGENET1K_V2: V2 uses a stronger training recipe than V1
    # (EMA, CutMix, etc.) → better feature representations to fine-tune from.
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features    # 2048 for ResNet-50
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# build_model() — Public Factory (Dependency Inversion Point)
# ──────────────────────────────────────────────────────────────────────────────

ModelType = Literal["custom_cnn", "resnet18", "resnet50"]


def build_model(model_type: ModelType = "resnet18", **kwargs) -> nn.Module:
    """
    Model factory — returns the requested architecture as an nn.Module.

    WHAT IT DOES:
      Acts as the single entry point to create any supported model.
      Callers (train.py, predict.py, the notebook) import only this
      function — they never import individual model classes.

    WHY A FACTORY FUNCTION (Dependency Inversion Principle):
      • Callers depend on the abstraction (build_model) not the
        concrete classes (CatDogCNN, get_resnet18_model).
      • Adding a new architecture in future only requires:
          1. Define a new function/class above.
          2. Add one line to the registry dict below.
        No change needed in train.py, predict.py, or the notebook.
      • Makes the codebase testable: tests mock build_model, not
        individual model constructors.

    WHY a dict registry instead of if/elif chain:
      • O(1) lookup vs O(n) branching.
      • Easier to inspect available models at runtime (list(registry)).
      • Clean error message with all valid options listed.

    Args:
        model_type: Architecture to build. One of:
                    'custom_cnn'  — 4-block CNN from scratch.
                    'resnet18'    — pretrained ResNet-18 (recommended).
                    'resnet50'    — pretrained ResNet-50 (highest accuracy).
        **kwargs:   Forwarded to the underlying constructor
                    (e.g., pretrained=False, freeze_backbone=True).
    Returns:
        Instantiated, ready-to-train PyTorch model (not yet on any device).
    Raises:
        ValueError: If model_type is not in the registry.
    """
    # WHY lambda wrappers: defers construction until the key is found,
    # avoiding unnecessary object creation for unused models.
    registry = {
        "custom_cnn": lambda: CatDogCNN(**kwargs),
        "resnet18":   lambda: get_resnet18_model(**kwargs),
        "resnet50":   lambda: get_resnet50_model(**kwargs),
    }

    if model_type not in registry:
        raise ValueError(
            f"Unknown model '{model_type}'. "
            f"Choose from: {list(registry.keys())}"
        )

    return registry[model_type]()


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test (run: python model.py)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    WHAT: Runs a quick smoke-test of all three model types.
    WHY: Catches shape mismatches and import errors before a full training run.
         Output: model name, output shape (should be (4,1)), and param count.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch  = torch.randn(4, 3, 224, 224).to(device)  # fake batch of 4 images

    for name in ("custom_cnn", "resnet18", "resnet50"):
        m      = build_model(name).to(device)
        out    = m(batch)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[{name:12s}] output={out.shape}  trainable_params={params:,}")
