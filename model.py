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

Three model options are provided:
  1. CatDogCNN   — Custom 4-block CNN (150×150 input, 2-class output)
  2. ResNet-18   — Transfer learning (~11M params, recommended)
  3. ResNet-50   — Transfer learning (~25M params, highest accuracy)

Usage:
    from model import build_model
    model = build_model('resnet18')   # recommended default
    model = build_model('custom_cnn') # trains from scratch, 150x150 input
    model = build_model('resnet50')   # maximum accuracy
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# CatDogCNN — Custom 4-Block Architecture (150×150 input)
# ──────────────────────────────────────────────────────────────────────────────

class CatDogCNN(nn.Module):
    """
    Custom CNN for cat/dog binary classification.

    Input:  (B, 3, 150, 150) — colour image resized to 150×150
    Output: (B, 2)           — raw logits [logit(Cat), logit(Dog)]

    Architecture (spatial sizes after each block):
      Block 1: Conv(3→32, k=3, pad=0)  + ReLU + MaxPool(2) → (B, 32,  74, 74)
      Block 2: Conv(32→64, k=3, pad=0) + ReLU + MaxPool(2) → (B, 64,  36, 36)
      Block 3: Conv(64→128,k=3, pad=0) + ReLU + MaxPool(2) → (B, 128, 17, 17)
      Block 4: Conv(128→128,k=3,pad=0) + ReLU + MaxPool(2) → (B, 128,  7,  7)
      Flatten                                               → (B, 6272)
      Linear(6272→512) + ReLU                               → (B, 512)
      Linear(512→2)                                         → (B, 2)  raw logits

    Loss: nn.CrossEntropyLoss — expects raw logits, applies log_softmax internally.
    Inference: apply torch.softmax(output, dim=1) to convert logits to probabilities.

    IMAGE_SIZE class attribute tells train.py / dataset.py which resize to use.
    """

    IMAGE_SIZE = 150   # expected input resolution

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 150×150 → 148×148 → 74×74
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2: 74×74 → 72×72 → 36×36
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3: 36×36 → 34×34 → 17×17
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 4: 17×17 → 15×15 → 7×7
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),   # 6272 → 512
            nn.ReLU(),
            nn.Linear(512, 2),              # 512  → 2 raw logits (no Softmax here)
            # NOTE: do NOT add Softmax here — CrossEntropyLoss applies log_softmax
            # internally and requires raw logits. Apply torch.softmax() at inference.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 150, 150) colour image batch.
        Returns:
            (B, 2) tensor — raw logits [logit(Cat), logit(Dog)].
            Use torch.softmax(output, dim=1) to convert to probabilities.
        """
        return self.classifier(self.features(x))


def build_model() -> nn.Module:
    """Instantiate and return the CatDogCNN model (not yet moved to any device)."""
    return CatDogCNN()


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test (run: python model.py)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch  = torch.randn(4, 3, 150, 150).to(device)
    model  = build_model().to(device)
    out    = model(batch)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CatDogCNN] output={out.shape}  trainable_params={params:,}")
    # expected: output=torch.Size([4, 2])  trainable_params=3,453,634
