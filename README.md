# Cats & Dogs Neural Network Classifier

Binary image classifier that identifies **Cat** or **Dog** in any photo using PyTorch transfer learning. Runs locally on a CUDA GPU or in Google Colab (free T4 GPU).

---

## Project Structure

```
L-39-CatsAndDogsNeuralNet/
├── CLAUDE.md                  ← Full task guide for the Claude agent
├── tasks.json                 ← Structured task list (T-001 – T-011)
├── requirements.txt           ← Python dependencies
├── model.py                   ← CNN model definitions (CustomCNN, ResNet-18/50)
├── dataset.py                 ← Oxford Pets data loading + augmentation
├── train.py                   ← GPU training script (main entry point)
├── predict.py                 ← Single-image inference
├── visualize.py               ← Visualization helpers (preview, curves, matrix, chart)
├── CatsAndDogs_Colab.ipynb    ← Google Colab end-to-end notebook
├── data/                      ← Auto-created — Oxford Pets dataset (~750 MB)
└── output/
    ├── best_model.pth         ← Best checkpoint (saved during training)
    ├── training_history.json  ← Per-epoch loss/accuracy log
    ├── training_curves.png    ← Loss & accuracy plot
    ├── data_preview.png       ← 2×2 sample image grid
    └── confusion_matrix.png   ← Validation confusion matrix
```

---

## Dataset

**Oxford-IIIT Pet Dataset** — public, free, auto-downloaded by torchvision.

| Property | Value |
|---|---|
| Source | `torchvision.datasets.OxfordIIITPet` |
| Total images | ~7,349 |
| Classes | Cat → `0`, Dog → `1` |
| Train / Val split | 80 / 20 (~5,879 / ~1,470) |
| Download size | ~750 MB |
| License | CC-BY-SA 4.0 |

No manual download needed — the dataset fetches itself on the first run.

---

## Setup

### Option A — Conda (recommended)

```bash
conda create -n catdog python=3.10 -y
conda activate catdog
pip install -r requirements.txt
```

### Option B — venv

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### Verify GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Install PyTorch with CUDA (if needed)

```bash
# Adjust the cu121 suffix to match your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Training

```bash
python train.py
```

Full options:

```bash
python train.py \
  --model      resnet18   \   # custom_cnn | resnet18 | resnet50
  --source     oxford     \   # oxford (auto-download) | local (your own data)
  --data-dir   ./data     \
  --epochs     15         \
  --batch-size 32         \
  --lr         0.0001     \
  --num-workers 4         \
  --output-dir ./output
```

### Model comparison

| Model | Params | Expected Val Acc | GPU time / epoch |
|---|---|---|---|
| `custom_cnn` | ~6 M | ~85–88 % | ~5 min |
| `resnet18` | ~11 M | ~92–95 % | ~2 min |
| `resnet50` | ~25 M | ~95–97 % | ~4 min |

---

## Inference

```bash
python predict.py --image path/to/cat.jpg
python predict.py --image path/to/dog.png --checkpoint output/best_model.pth
```

Example output:

```
=============================================
  Image : path/to/cat.jpg
  Result: Cat  (96.3% confident)
  P(Cat): 96.3%
  P(Dog):  3.7%
=============================================
```

---

## Visualization

### 1 — Data Preview (2×2 image grid, indices 6–9)

Displays four training images with their Cat/Dog labels as a dataset sanity check.

```bash
python visualize.py --mode preview
```

Output: `output/data_preview.png`

### 2 — Training Curves (loss & accuracy over epochs)

Reads `output/training_history.json` and plots train/val loss and accuracy side by side. The best validation epoch is marked with a dashed line.

```bash
python visualize.py --mode curves
```

Output: `output/training_curves.png`

### 3 — Confusion Matrix

Runs the trained model over the full validation split and renders a labelled 2×2 confusion matrix heatmap. Also prints precision, recall, and F1-score per class.

```bash
python visualize.py --mode confusion
```

Output: `output/confusion_matrix.png` and a classification report printed to the terminal.

### 4 — Unknown Image Prediction (image + confidence chart)

Accepts any JPEG/PNG image, runs inference, and displays the image alongside a horizontal bar chart showing Cat vs Dog confidence scores (green = predicted class, red = other).

```bash
python predict.py --image path/to/any_image.jpg
```

Output: `output/prediction_<imagename>.png` and a summary line in the terminal.

---

## Google Colab

1. Upload `CatsAndDogs_Colab.ipynb` to Google Drive.
2. Open with Google Colab (right-click → Open with → Colab).
3. Set GPU: `Runtime` → `Change Runtime Type` → `T4 GPU`.
4. Run all: `Runtime` → `Run all` (`Ctrl + F9`).

The notebook covers: GPU check → package install → data download → model definition → training loop → loss/accuracy plots → confusion matrix → sample predictions → model download.

---

## Architecture

### ResNet-18 (default)

```
ResNet-18 backbone (pretrained ImageNet weights)
  └── Final FC replaced with:
       Dropout(0.3) → Linear(512 → 1)
Output: raw logit → sigmoid → P(Dog)
```

### Custom CNN

```
Input: (B, 3, 224, 224)
  ConvBlock( 3 →  32) + MaxPool → (B,  32, 112, 112)
  ConvBlock(32 →  64) + MaxPool → (B,  64,  56,  56)
  ConvBlock(64 → 128) + MaxPool → (B, 128,  28,  28)
  ConvBlock(128→ 256) + MaxPool → (B, 256,  14,  14)
  AdaptiveAvgPool2d(4,4)        → (B, 256,   4,   4)
  Flatten → Linear(4096→512→128→1)
Output: raw logit
```

---

## Training Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Loss | `BCEWithLogitsLoss` | Numerically stable binary CE |
| Optimizer | `AdamW` | Weight decay 1e-4 |
| Scheduler | `CosineAnnealingLR` | Smooth LR decay |
| AMP | Enabled (CUDA only) | Mixed precision for speed |
| Grad clipping | max_norm = 1.0 | Prevents gradient explosion |
| Augmentation | Flip, Rotate, ColorJitter | Reduces overfitting |
| Input size | 224 × 224 | Standard ImageNet resolution |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA out of memory | `--batch-size 16` or `--model custom_cnn` |
| Dataset download fails | `--data-dir C:/Datasets/pets` |
| Windows DataLoader error | `--num-workers 0` |
| Slow training (CPU) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |

---

## Completion Checklist

- [ ] Environment setup complete
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] First training run completed (`output/best_model.pth` saved)
- [ ] `output/training_curves.png` generated
- [ ] Validation accuracy ≥ 90 %
- [ ] Data preview grid (`output/data_preview.png`) generated
- [ ] Confusion matrix (`output/confusion_matrix.png`) generated
- [ ] Unknown image prediction + confidence chart working
- [ ] Colab notebook runs end-to-end
