"""
app.py — Gradio Web UI for Cat vs Dog Prediction
=================================================
Launches a local web interface where you can upload any image and get:
  - The predicted class (Cat or Dog) with confidence
  - A confidence histogram showing P(Cat) and P(Dog)

Usage:
    python app.py
    python app.py --checkpoint output/best_model.pth --port 7860
    python app.py --share    # creates a temporary public URL (via Gradio)
"""

import argparse
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image

from model import build_model
from dataset import get_val_transform


CLASS_NAMES = ["Cat", "Dog"]
COLORS      = {"Cat": "#4C9BE8", "Dog": "#E87B4C"}


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """Load CatDogCNN from checkpoint. Falls back to random weights if missing."""
    model = build_model().to(device)
    ckpt_file = Path(checkpoint_path)
    if ckpt_file.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        val_acc = ckpt.get("val_acc", None)
        msg = f"val_acc={val_acc:.2f}%" if val_acc else "val_acc unknown"
        print(f"[Model] Loaded {checkpoint_path}  ({msg})")
    else:
        print(f"[Warning] Checkpoint not found: {checkpoint_path}  — using random weights")
    model.eval()
    return model


# ── Prediction function ───────────────────────────────────────────────────────

def make_predictor(model, device):
    """
    Returns a predict(image) function that:
      1. Runs the image through CatDogCNN.
      2. Returns a label dict for gr.Label  (class -> probability).
      3. Returns a matplotlib Figure        (confidence bar chart).
    """
    transform = get_val_transform(150)

    def predict(image: Image.Image):
        if image is None:
            return None, None

        # Preprocess
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(tensor)                              # (1, 2) raw logits

        probs    = torch.softmax(outputs[0], dim=0).cpu().tolist()  # [P(Cat), P(Dog)]
        prob_cat = probs[0]
        prob_dog = probs[1]
        predicted = CLASS_NAMES[int(prob_dog >= 0.5)]

        # ── Confidence histogram ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#0F1117")
        ax.set_facecolor("#1A1D27")

        values = [prob_cat * 100, prob_dog * 100]
        bars = ax.bar(
            CLASS_NAMES, values,
            color=[COLORS["Cat"], COLORS["Dog"]],
            edgecolor="white", linewidth=0.5, width=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.1f}%",
                ha="center", va="bottom",
                color="white", fontsize=14, fontweight="bold",
            )

        ax.set_ylim(0, 118)
        ax.set_ylabel("Confidence (%)", color="#AAA", fontsize=11)
        ax.set_title(
            f"Prediction: {predicted}  ({max(values):.1f}% confident)",
            color="white", fontsize=13, fontweight="bold", pad=10,
        )
        ax.tick_params(colors="#AAA")
        ax.grid(True, alpha=0.2, color="#555", axis="y")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        plt.tight_layout()

        # gr.Label expects {class_name: probability_0_to_1}
        label_dict = {CLASS_NAMES[0]: prob_cat, CLASS_NAMES[1]: prob_dog}
        return label_dict, fig

    return predict


# ── Gradio interface ──────────────────────────────────────────────────────────

def build_demo(predict_fn) -> gr.Blocks:
    with gr.Blocks(
        title="Cats vs Dogs Classifier",
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="orange",
        ),
        css=".output-label { font-size: 1.1rem; }",
    ) as demo:

        gr.Markdown(
            "## Cats vs Dogs Classifier\n"
            "Upload any photo — the **CatDogCNN** model predicts Cat or Dog "
            "and shows its confidence."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload image",
                    height=300,
                )
                run_btn = gr.Button("Predict", variant="primary")

            with gr.Column(scale=1):
                label_out = gr.Label(
                    num_top_classes=2,
                    label="Class probabilities",
                )
                plot_out = gr.Plot(label="Confidence histogram")

        run_btn.click(
            fn=predict_fn,
            inputs=image_input,
            outputs=[label_out, plot_out],
        )
        # Also trigger on image upload/change
        image_input.change(
            fn=predict_fn,
            inputs=image_input,
            outputs=[label_out, plot_out],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cats vs Dogs Gradio Web UI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="./output/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--port",       type=int, default=7860,
                        help="Local port to serve the UI on")
    parser.add_argument("--share",      action="store_true",
                        help="Create a temporary public Gradio URL")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model      = load_model(args.checkpoint, device)
    predict_fn = make_predictor(model, device)
    demo       = build_demo(predict_fn)

    print(f"[UI] Starting at http://localhost:{args.port}")
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
