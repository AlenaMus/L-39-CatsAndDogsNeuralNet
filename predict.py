"""
predict.py — Single-Image Inference
=====================================
Backend Developer Agent | Inference Layer | PyTorch

WHY THIS FILE EXISTS:
  Inference is a distinct concern from training (Single Responsibility).
  predict.py loads a saved checkpoint and classifies one image without
  importing any training machinery (no optimizer, no DataLoader, no AMP).
  This makes the inference path lightweight and deployable independently
  — for example, as a microservice or a batch script.

DESIGN PATTERN USED:
  CatDogPredictor follows the Service Object pattern:
    • Constructor: expensive setup (load model once).
    • predict(): cheap per-call inference (just a forward pass).
  This avoids re-loading model weights for every image when doing
  batch inference in a loop.

Usage:
    # Predict from a known path:
    python predict.py --image path/to/cat.jpg

    # Open a file-browser dialog to pick the image interactively:
    python predict.py --upload

    # Show confidence bar chart after prediction:
    python predict.py --image path/to/dog.png --chart

    # Upload + chart in one command:
    python predict.py --upload --chart
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image

from model import build_model
from dataset import get_val_transform, CLASS_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# browse_image() — Native File-Browser Dialog
# ──────────────────────────────────────────────────────────────────────────────

def browse_image() -> str:
    """
    Open a native OS file-browser dialog and return the selected image path.

    WHY TKINTER:
      tkinter ships with CPython on all platforms (Windows, macOS, Linux)
      and requires no extra installation. filedialog.askopenfilename()
      opens the OS-native file picker, which is familiar to users and
      filters for common image extensions automatically.

    WHY root.withdraw() + attributes('-topmost'):
      withdraw() hides the blank Tk root window so only the file dialog
      appears. '-topmost' ensures the dialog is not hidden behind other
      windows on Windows and macOS.

    FALLBACK:
      On headless servers (CI, SSH sessions) tkinter may be unavailable
      or display-less. The except block falls back to a plain stdin prompt
      so the script never hard-crashes.

    Returns:
        Absolute path string of the selected file, or '' if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()                        # hide blank root window
        root.attributes("-topmost", True)      # bring dialog to front
        path = filedialog.askopenfilename(
            title="Select a Cat or Dog image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files",   "*.*"),
            ],
        )
        root.destroy()
        return path or ""
    except Exception:
        return input("[Predict] Enter image path manually: ").strip()


# ──────────────────────────────────────────────────────────────────────────────
# show_confidence_chart() — Image + Bar Chart Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def show_confidence_chart(image_path: str, result: dict, save_dir: str = "./output") -> None:
    """
    Display the input image next to a horizontal Cat/Dog confidence bar chart
    and save the combined figure to disk.

    LAYOUT:
      Left subplot  — original image (un-normalised, as loaded from disk).
      Right subplot — horizontal barh chart:
                        • Green bar  = predicted (winning) class.
                        • Red bar    = other class.
                        • Dashed vertical line at 50 % (decision boundary).
                        • Percentage labels on each bar.

    WHY SAVE + SHOW:
      plt.savefig() is called before plt.show() so the file is written
      even if the display backend closes the window immediately (e.g. on
      some headless environments where show() is a no-op).

    Args:
        image_path: Path to the source image (used for display + filename).
        result:     Dict returned by CatDogPredictor.predict().
        save_dir:   Directory for the output PNG (created if absent).
    """
    img = Image.open(image_path).convert("RGB")

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Cat vs Dog — Prediction", fontsize=13, fontweight="bold")

    # ── Left: original image ──────────────────────────────────────────────────
    ax_img.imshow(img)
    ax_img.set_title("Input Image", fontsize=10)
    ax_img.axis("off")

    # ── Right: horizontal confidence bars ─────────────────────────────────────
    bar_labels = ["Cat", "Dog"]
    bar_values = [result["prob_cat"], result["prob_dog"]]
    pred       = result["label"]
    bar_colors = ["#2ecc71" if lbl == pred else "#e74c3c" for lbl in bar_labels]

    bars = ax_bar.barh(bar_labels, bar_values, color=bar_colors, height=0.4)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Confidence (%)", fontsize=10)
    ax_bar.set_title(
        f"Predicted: {pred}  ({result['confidence']:.1f}%)",
        fontsize=11, fontweight="bold",
    )
    ax_bar.axvline(x=50, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    for bar, val in zip(bars, bar_values):
        ax_bar.text(
            min(val + 1.5, 92),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", ha="left", fontsize=11, fontweight="bold",
        )

    plt.tight_layout()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    stem     = Path(image_path).stem
    out_path = Path(save_dir) / f"prediction_{stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Predict] Chart saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CatDogPredictor — Inference Service Object
# ──────────────────────────────────────────────────────────────────────────────

class CatDogPredictor:
    """
    Loads a trained model checkpoint and exposes a predict() method for
    single-image inference.

    WHAT IT DOES:
      1. __init__: loads the checkpoint once, reconstructs the model
         architecture, restores weights, switches to eval mode.
      2. predict(): preprocesses one image and returns a result dict.

    WHY A CLASS (not a plain function):
      Loading model weights from disk is slow (~0.5–2 seconds).
      If predict() were a standalone function it would reload weights on
      every call. Encapsulating in a class lets you:
        predictor = CatDogPredictor(...)   # load once
        for img_path in image_list:
            predictor.predict(img_path)   # reuse loaded model

    WHY map_location=self.device:
      A checkpoint saved on a CUDA GPU contains 'cuda:0' device references.
      If loaded on a CPU machine without map_location, PyTorch raises an
      error. map_location remaps tensors to the target device automatically,
      making the predictor portable across hardware environments.
    """

    def __init__(self, checkpoint_path: str, device: torch.device = None):
        """
        Load and prepare the model from a checkpoint file.

        WHAT IT DOES:
          Detects best available device, applies the deterministic val
          transform pipeline, loads the checkpoint dict, reconstructs the
          model architecture using the saved model_type key, restores
          state_dict, and sets the model to eval mode.

        WHY READ model_type FROM CHECKPOINT:
          The checkpoint saves which architecture was trained (resnet18,
          resnet50, custom_cnn). Without this, the predictor would need to
          be told the architecture separately — a brittle dependency.
          Storing it in the checkpoint makes the file self-describing.

        WHY model.eval():
          Switches BatchNorm layers from tracking per-batch statistics to
          using the running mean/variance accumulated during training.
          Also disables Dropout (which must not fire during inference).
          Without eval(), predictions would be non-deterministic and
          BatchNorm would give wrong activations on single images.

        Args:
            checkpoint_path: Path to a .pth file created by train.py.
            device: Override device (default: auto-detect CUDA/MPS/CPU).
        """
        # WHY auto-detect if no device provided: makes predictor usable
        # without the caller knowing or caring about GPU availability.
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Load the checkpoint dict (contains model_type, state_dict, metadata)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model_type = ckpt.get("model_type", "custom_cnn")

        self.transform   = get_val_transform(150)   # CatDogCNN uses 150×150
        self.class_names = CLASS_NAMES

        print(f"[Predict] Architecture : {model_type}")
        print(f"[Predict] Checkpoint   : {checkpoint_path}")
        print(f"[Predict] Val accuracy : {ckpt.get('val_acc', 'N/A'):.2f}%")

        # Reconstruct architecture WITHOUT pretrained weights (we have saved ones)
        self.model = build_model()
        self.model.load_state_dict(ckpt["model_state"])
        self.model = self.model.to(self.device)
        self.model.eval()   # critical: disable Dropout + use running BN stats

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        Predict whether a single image contains a Cat or Dog.

        WHAT IT DOES:
          1. Validates the file exists.
          2. Opens and converts to RGB (handles RGBA PNGs, greyscale, etc.).
          3. Applies the val transform pipeline to get a (3, 224, 224) tensor.
          4. Adds a batch dimension (.unsqueeze(0)) → (1, 3, 224, 224).
          5. Runs a forward pass through self.model.
          6. Converts logit to probability via sigmoid.
          7. Returns a human-readable result dict.

        WHY @torch.no_grad() DECORATOR:
          Disables autograd for the entire method. This:
            • Saves memory: no computation graph is built.
            • Speeds up inference: ~1.5× faster than with gradients.
            • Prevents accidental weight updates if someone mistakenly
              calls .backward() elsewhere.
          The decorator is preferred over a with-block because it applies
          to the entire method cleanly.

        WHY .convert("RGB"):
          Input images may be RGBA (PNG with transparency), greyscale (L),
          or even indexed palette images. The model expects exactly 3 channels.
          convert("RGB") normalises any PIL Image mode to a 3-channel format.

        WHY .unsqueeze(0) — adding a batch dimension:
          PyTorch models always expect a (B, C, H, W) 4D tensor.
          A single transformed image is (C, H, W) = (3, 224, 224).
          .unsqueeze(0) inserts a dimension at position 0 → (1, 3, 224, 224).
          This lets us reuse the exact same model forward pass as training
          without any code changes.

        WHY torch.sigmoid(logit):
          The model outputs a raw logit (unbounded scalar).
          sigmoid maps it to [0, 1] where:
            • Values ≥ 0.5 → Dog (positive class)
            • Values < 0.5 → Cat (negative class)
          This is the inverse of the log-odds transformation built into
          BCEWithLogitsLoss during training.

        WHY return a dict (not a tuple):
          A dict is self-documenting: the caller knows what 'label' and
          'confidence' mean without reading documentation. It is also
          extensible — adding 'breed' or 'bounding_box' in future does not
          change the function signature.

        Args:
            image_path: Path to a JPEG or PNG image file.
        Returns:
            dict with keys:
              'label'      — 'Cat' or 'Dog'
              'confidence' — confidence in the predicted class (%)
              'prob_cat'   — probability of Cat class (%)
              'prob_dog'   — probability of Dog class (%)
              'raw_logit'  — raw model output before sigmoid
        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image  = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        outputs  = self.model(tensor)                         # (1, 2) raw logits
        probs    = torch.softmax(outputs[0], dim=0).cpu()    # convert to probabilities
        prob_cat = probs[0].item()
        prob_dog = probs[1].item()

        predicted_class = "Dog" if prob_dog >= 0.5 else "Cat"
        confidence      = prob_dog if prob_dog >= 0.5 else prob_cat

        return {
            "label":      predicted_class,
            "confidence": round(confidence * 100, 2),
            "prob_cat":   round(prob_cat * 100, 2),
            "prob_dog":   round(prob_dog * 100, 2),
        }


# ──────────────────────────────────────────────────────────────────────────────
# parse_args() — Command-Line Interface
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Define and parse inference CLI arguments.

    WHAT IT DOES:
      Accepts two arguments:
        --image:      (required) path to the image to classify.
        --checkpoint: (optional) path to the .pth checkpoint file.

    WHY ARGPARSE FOR A SIMPLE SCRIPT:
      Even a two-argument script benefits from argparse because:
        • Auto-generates --help documentation.
        • Validates required arguments (--image) and reports clear errors.
        • Makes the script scriptable: shell scripts, CI, and automation
          tools can call predict.py without reading or modifying code.

    WHY DEFAULT CHECKPOINT PATH './output/best_model.pth':
      Matches the default output-dir of train.py so both scripts work
      together out-of-the-box without any configuration.

    Returns:
        argparse.Namespace with .image and .checkpoint attributes.
    """
    parser = argparse.ArgumentParser(
        description="Cats vs Dogs — Single Image Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the image file (JPEG or PNG). Omit when using --upload."
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help=(
            "Open a native file-browser dialog to select an image interactively. "
            "Takes precedence over --image. Automatically enables --chart."
        ),
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="Show and save a confidence bar chart alongside the input image.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./output/best_model.pth",
        help="Path to the trained model checkpoint (.pth file from train.py)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory where the confidence chart PNG is saved (default: ./output)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# main() — Entry Point for CLI Inference
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run inference on a single image and print results to stdout.

    WHAT IT DOES:
      1. Parse CLI arguments.
      2. Instantiate CatDogPredictor (loads model once).
      3. Call predictor.predict() on the specified image.
      4. Print a formatted result block.

    WHY SEPARATE main() FROM CatDogPredictor:
      main() is UI logic (CLI parsing + printing).
      CatDogPredictor is business logic (model loading + inference).
      Keeping them separate means CatDogPredictor can be imported into
      a web API, a Desktop app, or a Jupyter notebook without any CLI code.

    WHY PRINT BOTH PROBABILITIES:
      Showing P(Cat) and P(Dog) separately gives the user full insight
      into the model's confidence. A 60% Dog prediction is far less
      reliable than a 99% Dog prediction — printing both helps the user
      decide whether to trust or double-check the result.

    WHY --upload IMPLIES --chart:
      When a user browses to select an image interactively they almost
      always want to see the visual result immediately. Forcing them to
      also pass --chart would be an unnecessary extra step.
    """
    args = parse_args()

    # ── Resolve image path ────────────────────────────────────────────────────
    if args.upload:
        print("[Predict] Opening file browser — select a Cat or Dog image ...")
        image_path = browse_image()
        if not image_path:
            print("[Predict] No file selected. Exiting.")
            return
        print(f"[Predict] Selected : {image_path}")
    elif args.image:
        image_path = args.image
    else:
        print("[Predict] Error: supply --image <path> or use --upload to browse.")
        return

    # ── Run inference ─────────────────────────────────────────────────────────
    predictor = CatDogPredictor(checkpoint_path=args.checkpoint)
    result    = predictor.predict(image_path)

    print("\n" + "=" * 45)
    print(f"  Image  : {image_path}")
    print(f"  Result : {result['label']}  ({result['confidence']:.1f}% confident)")
    print(f"  P(Cat) : {result['prob_cat']:.1f}%")
    print(f"  P(Dog) : {result['prob_dog']:.1f}%")
    print("=" * 45)

    # ── Confidence chart (always shown when --upload; optional with --chart) ──
    if args.chart or args.upload:
        show_confidence_chart(image_path, result, save_dir=args.output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    WHAT: Standard Python entry guard — only runs when called as a script.
    WHY: Allows predict.py to be imported in tests or other scripts
         (e.g., a batch inference loop) without triggering CLI parsing.
    """
    main()
