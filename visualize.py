"""
visualize.py — Post-Training Visualization Suite
=================================================
Generates rich visual reports from training history and model predictions.

Outputs saved to the output/ directory:
  - training_curves.png         (already made by train.py — enhanced version here)
  - confusion_matrix.png        — TP/TN/FP/FN heatmap on the validation set
  - sample_predictions.png      — 24-image grid with confidence scores
  - confidence_distribution.png — histogram of model certainty (Cat vs Dog)
  - per_class_accuracy.png      — bar chart of Cat vs Dog accuracy

Usage:
    python visualize.py
    python visualize.py --checkpoint output/best_model.pth --output-dir output
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix

from model import build_model
from dataset import get_dataloaders, get_val_transform

CLASS_NAMES = ["Cat", "Dog"]
COLORS = {"Cat": "#4C9BE8", "Dog": "#E87B4C"}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint, returns (model, val_acc)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    val_acc = ckpt.get("val_acc", None)
    print(f"[Checkpoint] Loaded CatDogCNN — best val acc: {val_acc:.2f}%")
    return model, val_acc


@torch.no_grad()
def collect_predictions(model, val_loader, device):
    """
    Run model over the validation set and collect:
      - all_probs:  P(Dog) for each image  [N]   — Softmax output[:,1]
      - all_labels: true binary labels      [N]   — 0=Cat, 1=Dog
      - all_images: raw tensors (still normalised) [N, 3, H, W]
    """
    all_probs, all_labels, all_images = [], [], []
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)[:, 1].cpu()  # P(Dog) after softmax
        all_probs.append(probs)
        all_labels.append(labels)
        all_images.append(images.cpu())
        if len(all_probs) * images.size(0) >= 1500:
            break
    return (
        torch.cat(all_probs).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_images),
    )


def denormalize(tensor):
    """Reverse ImageNet normalisation so images display correctly."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor * std + mean
    return img.permute(1, 2, 0).clamp(0, 1).numpy()


# ── plot 1: enhanced training curves ─────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str) -> None:
    """Enhanced version: adds best-epoch marker and shaded overfitting gap."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    best_ep = int(np.argmax(history["val_acc"])) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0F1117")
    for ax in axes:
        ax.set_facecolor("#1A1D27")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="#4C9BE8", lw=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   color="#E87B4C", lw=2, label="Val Loss")
    ax.fill_between(epochs, history["train_loss"], history["val_loss"],
                    alpha=0.15, color="#E87B4C", label="Gap")
    ax.axvline(best_ep, color="#FFD700", ls="--", lw=1.5, label=f"Best (ep {best_ep})")
    ax.set_title("Loss Curve", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Epoch", color="#AAA"); ax.set_ylabel("BCE Loss", color="#AAA")
    ax.tick_params(colors="#AAA"); ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2, color="#555")

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], color="#4C9BE8", lw=2, label="Train Acc")
    ax.plot(epochs, history["val_acc"],   color="#E87B4C", lw=2, label="Val Acc")
    ax.axvline(best_ep, color="#FFD700", ls="--", lw=1.5, label=f"Best (ep {best_ep})")
    best_val = max(history["val_acc"])
    ax.axhline(best_val, color="#50FA7B", ls=":", lw=1.2, label=f"Best val {best_val:.1f}%")
    ax.set_title("Accuracy Curve", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Epoch", color="#AAA"); ax.set_ylabel("Accuracy (%)", color="#AAA")
    ax.tick_params(colors="#AAA"); ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2, color="#555")

    plt.suptitle("Cats vs Dogs — Training Progress", color="white",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] training_curves_enhanced.png -> {save_path}")


# ── plot 2: confusion matrix ──────────────────────────────────────────────────

def plot_confusion_matrix(probs, labels, save_path: str) -> None:
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels.astype(int), preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor="#333",
                annot_kws={"size": 18, "weight": "bold"}, ax=ax)

    ax.set_xlabel("Predicted", color="white", fontsize=12)
    ax.set_ylabel("Actual",    color="white", fontsize=12)
    ax.set_title("Confusion Matrix — Validation Set", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), color="white")
    plt.setp(ax.get_yticklabels(), color="white", rotation=0)

    # Annotate TN/TP/FP/FN
    labels_map = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.78, labels_map[i][j],
                    ha="center", va="center", color="#FFD700",
                    fontsize=10, fontweight="bold")

    acc = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100
    fig.text(0.5, -0.02, f"Overall Accuracy: {acc:.2f}%", ha="center",
             color="#50FA7B", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] confusion_matrix.png -> {save_path}")


# ── plot 3: sample predictions grid ──────────────────────────────────────────

def plot_sample_predictions(images, probs, labels, save_path: str, n=9) -> None:
    """3x3 grid of sample images with true/predicted labels and confidence score."""
    n = min(n, len(probs))
    # Select a balanced mix: some correct, some incorrect (or all if not enough)
    correct_idx   = np.where((probs >= 0.5).astype(int) == labels.astype(int))[0]
    incorrect_idx = np.where((probs >= 0.5).astype(int) != labels.astype(int))[0]
    n_wrong  = min(n // 3, len(incorrect_idx))
    n_right  = n - n_wrong
    chosen   = np.concatenate([
        np.random.choice(correct_idx,   n_right, replace=False),
        np.random.choice(incorrect_idx, n_wrong, replace=False) if n_wrong > 0 else [],
    ]).astype(int)
    np.random.shuffle(chosen)

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.6))
    fig.patch.set_facecolor("#0F1117")
    axes = axes.flatten()

    for idx, ax in zip(chosen, axes):
        img   = denormalize(images[idx])
        prob  = float(probs[idx])
        true  = int(labels[idx])
        pred  = 1 if prob >= 0.5 else 0
        conf  = prob if pred == 1 else 1 - prob
        correct = (pred == true)

        ax.imshow(img)
        ax.axis("off")
        border_color = "#50FA7B" if correct else "#FF5555"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

        title_color = "#50FA7B" if correct else "#FF5555"
        ax.set_title(
            f"{'[OK]' if correct else '[X]'} {CLASS_NAMES[pred]} ({conf*100:.0f}%)\nTrue: {CLASS_NAMES[true]}",
            color=title_color, fontsize=8, pad=3
        )

    for ax in axes[len(chosen):]:
        ax.axis("off")

    plt.suptitle("Sample Predictions — Green=Correct  Red=Wrong",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] sample_predictions.png -> {save_path}")


# ── plot 4: confidence distribution ──────────────────────────────────────────

def plot_confidence_distribution(probs, labels, save_path: str) -> None:
    """Histogram of P(Dog) separately for true Cats and true Dogs."""
    cat_probs = probs[labels == 0]
    dog_probs = probs[labels == 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")

    bins = np.linspace(0, 1, 40)
    ax.hist(cat_probs, bins=bins, alpha=0.7, color="#4C9BE8",
            label=f"True Cat (n={len(cat_probs)})", edgecolor="white", lw=0.3)
    ax.hist(dog_probs, bins=bins, alpha=0.7, color="#E87B4C",
            label=f"True Dog (n={len(dog_probs)})", edgecolor="white", lw=0.3)
    ax.axvline(0.5, color="#FFD700", ls="--", lw=2, label="Decision boundary (0.5)")

    ax.set_xlabel("P(Dog)  ->  0 = certain Cat  |  1 = certain Dog",
                  color="#AAA", fontsize=11)
    ax.set_ylabel("Count", color="#AAA")
    ax.set_title("Confidence Distribution on Validation Set",
                 color="white", fontsize=14, pad=12)
    ax.tick_params(colors="#AAA")
    ax.legend(framealpha=0.3, labelcolor="white")
    ax.grid(True, alpha=0.2, color="#555")

    # Shade regions
    ax.axvspan(0, 0.5, alpha=0.05, color="#4C9BE8")
    ax.axvspan(0.5, 1, alpha=0.05, color="#E87B4C")
    ax.text(0.25, ax.get_ylim()[1] * 0.95, "Cat Zone", color="#4C9BE8",
            ha="center", fontsize=10, alpha=0.8)
    ax.text(0.75, ax.get_ylim()[1] * 0.95, "Dog Zone", color="#E87B4C",
            ha="center", fontsize=10, alpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] confidence_distribution.png -> {save_path}")


# ── plot 5: per-class accuracy ────────────────────────────────────────────────

def plot_per_class_accuracy(probs, labels, save_path: str) -> None:
    """Bar chart: Cat accuracy, Dog accuracy, overall accuracy."""
    preds  = (probs >= 0.5).astype(int)
    cat_acc = (preds[labels == 0] == 0).mean() * 100
    dog_acc = (preds[labels == 1] == 1).mean() * 100
    overall = (preds == labels.astype(int)).mean() * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")

    bars = ax.bar(["Cat Accuracy", "Dog Accuracy", "Overall"],
                  [cat_acc, dog_acc, overall],
                  color=["#4C9BE8", "#E87B4C", "#50FA7B"],
                  edgecolor="white", linewidth=0.5, width=0.5)

    for bar, val in zip(bars, [cat_acc, dog_acc, overall]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                color="white", fontsize=13, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", color="#AAA")
    ax.set_title("Per-Class Accuracy — Validation Set",
                 color="white", fontsize=14, pad=12)
    ax.tick_params(colors="#AAA")
    ax.grid(True, alpha=0.2, color="#555", axis="y")
    ax.axhline(90, color="#FFD700", ls="--", lw=1, alpha=0.5, label="90% target")
    ax.legend(framealpha=0.3, labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] per_class_accuracy.png -> {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Enhanced training curves from saved history
    history_path = out / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, str(out / "training_curves_enhanced.png"))
    else:
        print(f"[Skip] {history_path} not found — skipping curve plots")

    # 2. Load model + run validation for prediction-based plots
    ckpt_path = out / "best_model.pth"
    if not ckpt_path.exists():
        print(f"[Skip] {ckpt_path} not found — skipping prediction plots")
        return

    model, _ = load_checkpoint(str(ckpt_path), device)

    print("[Data] Loading validation set …")
    _, val_loader, _ = get_dataloaders(
        source="oxford", data_dir=args.data_dir,
        batch_size=64, num_workers=0, image_size=150,
    )

    print("[Inference] Collecting predictions …")
    probs, labels, images = collect_predictions(model, val_loader, device)

    plot_confusion_matrix(probs, labels,
                          str(out / "confusion_matrix.png"))
    plot_sample_predictions(images, probs, labels,
                            str(out / "sample_predictions.png"), n=9)
    plot_confidence_distribution(probs, labels,
                                 str(out / "confidence_distribution.png"))
    plot_per_class_accuracy(probs, labels,
                            str(out / "per_class_accuracy.png"))

    print(f"\n[Done] All visualizations saved to {out}/")
    for f in sorted(out.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate post-training visualization plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",  type=str, default="./output/best_model.pth")
    parser.add_argument("--data-dir",    type=str, default="./data")
    parser.add_argument("--output-dir",  type=str, default="./output")
    main(parser.parse_args())
