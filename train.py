"""
train.py — GPU Training Script (Main Entry Point)
==================================================
Backend Developer Agent | PyTorch | CUDA / MPS / CPU

WHY THIS FILE EXISTS:
  This is the orchestrator. It wires together the model (model.py),
  the data (dataset.py), the loss, optimizer, and scheduler into a
  complete training loop. It follows the Service Layer pattern:
  each concern (device setup, one epoch, validation, plotting) is its
  own function so the main() function reads like a high-level recipe.

DESIGN PRINCIPLES APPLIED:
  • Single Responsibility — each function does exactly one thing.
  • Separation of Concerns — data, model, training logic are fully decoupled.
  • Fail Fast — device and data problems surface at startup, not mid-run.
  • Observability — loss/accuracy logged every epoch; history saved as JSON.
  • Recoverability — best checkpoint saved whenever val accuracy improves.

Usage:
    python train.py                            # ResNet-18, Oxford Pets, 15 epochs
    python train.py --model resnet50 --epochs 30
    python train.py --source local --data-dir ./data
    python train.py --help
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from model import build_model, ModelType
from dataset import get_dataloaders


# ──────────────────────────────────────────────────────────────────────────────
# get_device() — Hardware Selection
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Auto-detect and return the best available compute device.

    WHAT IT DOES:
      Checks for CUDA (NVIDIA GPU), then MPS (Apple Silicon GPU), then
      falls back to CPU. Prints hardware details so the user knows
      immediately whether GPU training is active.

    WHY THE PRIORITY ORDER (CUDA > MPS > CPU):
      1. CUDA: NVIDIA GPUs provide the broadest PyTorch operation support and
         the highest throughput for image training. CUDA also enables
         Automatic Mixed Precision (AMP) — not supported on MPS or CPU.
      2. MPS: Apple's Metal GPU backend (M1/M2/M3 Macs). Meaningful speedup
         over CPU but lacks AMP support and some CUDA ops.
      3. CPU: Fallback guaranteeing the code runs anywhere, albeit slowly.

    WHY PRINT GPU NAME AND VRAM:
      Users often have multiple GPUs or share a machine. Showing the device
      name and available VRAM helps diagnose OOM errors immediately
      (if VRAM < 4 GB, reduce batch size before training starts).

    Returns:
        torch.device — CUDA, MPS, or CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[GPU] Apple MPS (Metal Performance Shaders) enabled.")
    else:
        device = torch.device("cpu")
        print("[CPU] No GPU found — using CPU. Training will be slow.")
    return device


# ──────────────────────────────────────────────────────────────────────────────
# train_one_epoch() — One Training Epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    """
    Train the model for one complete pass over the training DataLoader.

    WHAT IT DOES:
      Iterates over every batch, performs a forward pass, computes the loss,
      back-propagates gradients, clips them, and updates the weights.
      Tracks running loss and correct predictions to return epoch statistics.

    WHY model.train() AT THE START:
      Puts BatchNorm and Dropout layers in training mode.
      • BatchNorm uses per-batch statistics (not running averages).
      • Dropout randomly zeroes neurons to prevent co-adaptation.
      Both are disabled during validation via model.eval().

    WHY optimizer.zero_grad(set_to_none=True):
      Clears accumulated gradients before each batch.
      set_to_none=True sets gradient tensors to None instead of zeros,
      which saves memory and is slightly faster — recommended since PyTorch 1.7.

    WHY autocast (Automatic Mixed Precision):
      AMP runs the forward pass in float16 where safe (most operations)
      and float32 where precision is critical (e.g. loss computation).
      This roughly halves VRAM usage and speeds training by 1.5–2×
      on modern NVIDIA GPUs with Tensor Cores.
      WHY enabled only on CUDA: MPS and CPU do not support float16 AMP.

    WHY GradScaler:
      AMP's float16 has a smaller dynamic range than float32. GradScaler
      multiplies the loss by a large factor before backward() and then
      divides gradients back before the update — preventing gradients
      from becoming zero ('underflow') in float16.
      The scale factor adapts automatically: increases after stable steps,
      decreases after NaN/inf gradients ('overflow').

    WHY clip_grad_norm_(max_norm=1.0):
      Gradient clipping prevents exploding gradients, which can crash
      training with NaN weights. max_norm=1.0 is a conservative threshold
      suitable for most image classification tasks. It clips the L2-norm
      of all gradient tensors to 1.0 without changing their direction.

    WHY BCEWithLogitsLoss (through criterion):
      Binary Cross-Entropy with Logits fuses sigmoid + BCE in a single
      numerically-stable computation (log-sum-exp trick).
      Using torch.sigmoid().then nn.BCELoss() separately is less stable
      and slightly slower. This is the standard for binary classification.

    Args:
        model:     Neural network — must be on the correct device.
        loader:    Training DataLoader.
        criterion: Loss function (BCEWithLogitsLoss).
        optimizer: AdamW optimiser instance.
        scaler:    GradScaler for AMP (no-op if use_amp=False).
        device:    torch.device to move tensors to.
        use_amp:   Enable mixed precision (True only on CUDA).
    Returns:
        (avg_loss, accuracy_percent) — average over the full epoch.
    """
    model.train()  # activate BatchNorm + Dropout training behaviour
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        # Move to GPU — non_blocking=True lets CPU continue while GPU copies
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)  # (B,) → (B,1)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass inside autocast for AMP
        with autocast("cuda", enabled=use_amp):
            outputs = model(images)           # (B, 1) raw logits
            loss    = criterion(outputs, labels)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)            # unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)                # updates weights (if no NaN/inf)
        scaler.update()                       # adjusts scale factor for next step

        # Accumulate metrics
        running_loss += loss.item() * images.size(0)  # weight by batch size
        preds   = (torch.sigmoid(outputs) >= 0.5).float()  # logit → binary pred
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ──────────────────────────────────────────────────────────────────────────────
# validate() — Validation Epoch
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set (no weight updates).

    WHAT IT DOES:
      Passes validation images through the model, computes loss and accuracy,
      but performs no back-propagation. Returns per-epoch statistics.

    WHY @torch.no_grad() DECORATOR:
      Disables gradient tracking for every tensor operation inside this
      function. Benefits:
        1. Reduces memory: no computation graph is stored.
        2. Faster: PyTorch skips gradient bookkeeping (~1.5× faster).
        3. Prevents accidental weight updates.
      Using the decorator (rather than a with-block inside the loop) is
      cleaner and ensures no gradient leaks if someone adds code later.

    WHY model.eval() IS CALLED BY THE CALLER (main):
      We call model.eval() before validate() to switch BatchNorm to
      use its running statistics (not batch statistics) and disable Dropout.
      WHY here: eval() is called at the start of validate() to ensure
      correctness even if the function is called standalone.

    WHY SIGMOID THRESHOLD 0.5:
      sigmoid(0) = 0.5. A logit > 0 means P(Dog) > 0.5 → predict Dog.
      0.5 is the natural decision boundary for balanced classes.
      For imbalanced data you would tune this threshold on a dev set.

    Args:
        model:     Model in eval mode.
        loader:    Validation DataLoader.
        criterion: Loss function (same as training for fair comparison).
        device:    Compute device.
    Returns:
        (avg_loss, accuracy_percent) over the entire validation set.
    """
    model.eval()  # disable Dropout; BatchNorm uses running statistics
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds   = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ──────────────────────────────────────────────────────────────────────────────
# save_training_plot() — Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def save_training_plot(history: Dict, save_path: str) -> None:
    """
    Generate and save training loss and accuracy curves as a PNG image.

    WHAT IT DOES:
      Creates a 2-panel matplotlib figure:
        Left panel:  Train vs Val loss over epochs.
        Right panel: Train vs Val accuracy over epochs.
      Saves the figure to disk and closes it to free memory.

    WHY SAVE CURVES TO DISK:
      In headless environments (remote servers, cloud VMs, Colab detached
      sessions) plt.show() has no effect. Saving to PNG ensures results
      are always accessible regardless of the runtime environment.

    WHY BOTH LOSS AND ACCURACY PANELS:
      Loss shows granular optimisation progress (smoother, more informative
      for debugging). Accuracy shows the business metric that determines
      model quality. Together they reveal:
        • Overfitting: train accuracy >> val accuracy.
        • Underfitting: both plateau early at low accuracy.
        • Good fit: train and val curves close, both high.

    WHY alpha=0.4 GRID:
      A faint grid aids readability without competing visually with the
      data lines.

    Args:
        history:   Dict with keys 'train_loss', 'val_loss',
                   'train_acc', 'val_acc'. Each a list of epoch values.
        save_path: Absolute or relative path to write the PNG file.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Left panel — Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    axes[0].set_title("Loss Curve", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # Right panel — Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc",   markersize=4)
    axes[1].set_title("Accuracy Curve", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.suptitle("Cats vs Dogs — Training Progress", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()   # WHY close: prevents memory leak if train.py is imported as a module
    print(f"[Plot] Saved training curves -> {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# main() — Training Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """
    Orchestrate the full training pipeline from raw arguments to saved model.

    WHAT IT DOES:
      1. Detect device (get_device).
      2. Build DataLoaders (get_dataloaders).
      3. Instantiate the model (build_model).
      4. Wire up loss, optimiser, scheduler, and scaler.
      5. Run the epoch loop: train → validate → checkpoint if improved.
      6. Save training history (JSON) and curves (PNG).

    WHY SEPARATE main() FROM THE EPOCH LOGIC:
      main() is a coordinator — it knows the order of operations but
      delegates each step to single-purpose functions. This makes each
      step independently testable and readable at a glance.

    WHY AdamW (not plain Adam or SGD):
      • Adam: adaptive learning rates per parameter — faster convergence
        than SGD, especially with transfer learning where different
        layers need different effective learning rates.
      • AdamW ('decoupled weight decay'): fixes Adam's incorrect weight
        decay implementation (Loshchilov & Hutter, 2017). With AdamW,
        weight_decay acts as proper L2 regularisation, improving
        generalisation on medium-sized datasets like Oxford Pets.
      WHY lr=1e-4: conservative enough to fine-tune pretrained weights
        without destroying learned features. 1e-3 would be too aggressive.

    WHY CosineAnnealingLR (not StepLR or ReduceLROnPlateau):
      Cosine annealing decays the learning rate smoothly from lr to eta_min
      following a cosine curve. Benefits vs alternatives:
        • StepLR: drops LR by a fixed factor every N epochs — abrupt jumps
          can destabilise training.
        • ReduceLROnPlateau: reactive (only drops on stagnation) — less
          predictable and requires careful patience tuning.
        • CosineAnnealingLR: continuous smooth decay → model stays in a
          region of the loss landscape with good generalisation. Widely used
          in SOTA image models (ResNet, ViT, EfficientNet papers).

    WHY SAVE BEST MODEL (not last model):
      Late-epoch overfitting is common. The last epoch's model may have
      higher training accuracy but worse generalisation than an earlier
      checkpoint. Saving on best val accuracy captures the optimal generalisation.

    WHY save history as JSON:
      JSON is human-readable, easily imported into pandas/numpy for analysis,
      and compatible with all tools. Saving training data enables offline
      analysis without re-training.

    Args:
        args: Parsed argparse.Namespace with all CLI arguments.
    """
    print("=" * 60)
    print("  Cats vs Dogs Neural Network Trainer")
    print("  Backend Developer Agent | PyTorch")
    print("=" * 60)

    device  = get_device()
    use_amp = device.type == "cuda"  # AMP only supported on CUDA

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, class_names = get_dataloaders(
        source=args.source,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[Model] Building '{args.model}' ...")
    model = build_model(args.model, pretrained=True)
    model = model.to(device)   # move all parameters/buffers to GPU
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {n_params:,}")

    # ── Loss function ─────────────────────────────────────────────────────────
    # WHY BCEWithLogitsLoss: numerically stable sigmoid + BCE in one op
    criterion = nn.BCEWithLogitsLoss()

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # WHY weight_decay=1e-4: mild L2 regularisation to reduce overfitting
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── LR Scheduler ────────────────────────────────────────────────────────
    # WHY T_max=epochs: completes one full cosine cycle over the run
    # eta_min=1e-6: never drops to zero — keeps gradient signal alive
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── AMP Scaler ───────────────────────────────────────────────────────────
    # enabled=False on CPU/MPS makes it a transparent no-op wrapper
    scaler = GradScaler("cuda", enabled=use_amp)

    # ── Training Setup ───────────────────────────────────────────────────────
    history: Dict = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc  = 0.0
    save_dir      = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt     = save_dir / "best_model.pth"

    print(f"\n[Train] Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  AMP: {use_amp}")
    print(f"[Train] Checkpoints -> {save_dir}")
    print("-" * 60)

    # ── Epoch Loop ───────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()  # advance cosine LR schedule after each epoch

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save checkpoint if this is the best model so far
        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch,
                "model_type":  args.model,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "class_names": class_names,
            }, best_ckpt)
            flag = "  * BEST"

        print(
            f"Epoch [{epoch:>3}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}%  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%  "
            f"({elapsed:.1f}s){flag}"
        )

    # ── Post-training ────────────────────────────────────────────────────────
    print("-" * 60)
    print(f"[Done] Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"[Done] Model saved -> {best_ckpt}")

    # Save history for offline analysis / plotting
    history_path = save_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Done] History saved -> {history_path}")

    # Generate and save training plots
    save_training_plot(history, str(save_dir / "training_curves.png"))


# ──────────────────────────────────────────────────────────────────────────────
# parse_args() — Command-Line Interface
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Define and parse command-line arguments.

    WHAT IT DOES:
      Uses argparse to define all configurable hyperparameters and paths
      as CLI flags with sensible defaults. Returns a Namespace object.

    WHY ARGPARSE (not hardcoded constants or a config file):
      • Allows different experiments without editing source code.
      • Self-documenting: python train.py --help shows all options.
      • Scriptable: CI/CD and hyperparameter search tools can call
        train.py with different flags without code changes.
      • DefaultsHelpFormatter: automatically shows default values in --help.

    WHY THESE SPECIFIC DEFAULTS:
      --model resnet18:   Best accuracy/speed trade-off for a single GPU.
      --source oxford:    Auto-downloads data; no manual steps needed.
      --epochs 15:        Sufficient for ResNet-18 to reach ~92% val acc
                          on the Oxford dataset.
      --batch-size 32:    Fits in 4–6 GB VRAM; safe for most consumer GPUs.
      --lr 1e-4:          Conservative LR for fine-tuning pretrained weights.
      --num-workers 4:    Overlaps data loading with GPU computation.

    Returns:
        argparse.Namespace — attribute access to all CLI flag values.
    """
    parser = argparse.ArgumentParser(
        description="Cats vs Dogs Neural Net Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",       type=str, default="resnet18",
                        choices=["custom_cnn", "resnet18", "resnet50"],
                        help="Model architecture to train")
    parser.add_argument("--source",      type=str, default="oxford",
                        choices=["oxford", "local"],
                        help="Dataset source: 'oxford' auto-downloads, 'local' uses data/train+val")
    parser.add_argument("--data-dir",    type=str, default="./data",
                        help="Root directory for dataset storage")
    parser.add_argument("--epochs",      type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch-size",  type=int, default=32,
                        help="Batch size (reduce to 16 if VRAM < 4 GB)")
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Initial learning rate for AdamW")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader prefetch worker processes (use 0 on Windows if errors)")
    parser.add_argument("--output-dir",  type=str, default="./output",
                        help="Directory to save checkpoints, history, and plots")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    WHAT: Standard Python entry point — only runs when called directly.
    WHY '__main__' guard: allows train.py to be imported by tests or other
    scripts without triggering training. Essential for testability.
    """
    args = parse_args()
    main(args)
