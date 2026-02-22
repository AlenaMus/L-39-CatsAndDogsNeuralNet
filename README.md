# Cats & Dogs Neural Network Classifier

Binary image classifier that identifies **Cat** or **Dog** in any photo using PyTorch transfer learning. Runs locally on a CUDA GPU or in Google Colab (free T4 GPU).

---

## Training Results

> ResNet-18 fine-tuned on Oxford-IIIT Pet — trained on NVIDIA GeForce RTX 5080 (16 GB VRAM)

| Metric | Value |
|---|---|
| Best Validation Accuracy | **99.73 %** (epoch 14) |
| Final Train Accuracy | 100.0 % |
| Final Val Accuracy | 99.59 % |
| Final Train Loss | 0.0019 |
| Final Val Loss | 0.0071 |
| Time per Epoch | ~24 s |
| Total Training Time | ~6 min (15 epochs) |

### Per-Epoch Log

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.1363 | 93.9 % | 0.0239 | 99.3 % |
| 2 | 0.0373 | 98.7 % | 0.0163 | 99.5 % |
| 3 | 0.0301 | 98.8 % | 0.0366 | 98.8 % |
| 4 | 0.0220 | 99.3 % | 0.0236 | 98.9 % |
| 5 | 0.0115 | 99.7 % | 0.0256 | 99.0 % |
| 6 | 0.0114 | 99.7 % | 0.0119 | 99.5 % |
| 7 | 0.0149 | 99.6 % | 0.0340 | 98.6 % |
| 8 | 0.0086 | 99.7 % | 0.0156 | 99.3 % |
| 9 | 0.0030 | 99.9 % | 0.0120 | 99.5 % |
| 10 | 0.0045 | 99.9 % | 0.0130 | 99.5 % |
| 11 | 0.0036 | 99.9 % | 0.0082 | 99.6 % |
| 12 | 0.0029 | 99.9 % | 0.0123 | 99.6 % |
| 13 | 0.0039 | 99.8 % | 0.0072 | 99.6 % |
| **14** | **0.0016** | **99.9 %** | **0.0076** | **99.7 % *** |
| 15 | 0.0019 | 100.0 % | 0.0071 | 99.6 % |

`* = best checkpoint saved`

---

## Validation / Testing Results

> Best checkpoint (`best_model.pth`, epoch 14) evaluated on the held-out validation set — 736 images never seen during training.

### Per-Class Accuracy

| Class | Correct | Total | Accuracy |
|---|---|---|---|
| Cat | 240 | 240 | **100.00 %** |
| Dog | 494 | 496 | **99.60 %** |
| **Overall** | **734** | **736** | **99.73 %** |

### Confusion Matrix

|  | Predicted Cat | Predicted Dog |
|---|---|---|
| **Actual Cat** | 240 (TN) | 0 (FP) |
| **Actual Dog** | 2 (FN) | 494 (TP) |

- **0 false positives** — every image the model called "Cat" was truly a Cat.
- **2 false negatives** — 2 Dog images were misclassified as Cat (out of 496 Dogs).

### Inference Results — Custom Images

| Image | Result | P(Cat) | P(Dog) | Chart |
|---|---|---|---|---|
| `cat-pet-animal-domestic-104827.jpeg` | **Cat** | **100.0 %** | 0.0 % | `output/prediction_cat-pet-animal-domestic-104827.png` |
| `hero-image1.jpg` | **Dog** | 0.0 % | **100.0 %** | `output/prediction_hero-image1.png` |

### Output Files

```
output/
├── best_model.pth                                    (43 MB) — ResNet-18 weights, 99.73% val acc
├── training_history.json                             (1.6 KB) — per-epoch loss/accuracy log
├── training_curves.png                               (95 KB) — loss & accuracy curves
├── training_curves_enhanced.png                      (121 KB) — dark-theme curves with best-epoch marker
├── confusion_matrix.png                              (39 KB) — TP/TN/FP/FN heatmap
├── sample_predictions.png                            (3.9 MB) — 24-image prediction grid
├── confidence_distribution.png                       (54 KB) — P(Dog) histogram by true class
├── per_class_accuracy.png                            (42 KB) — Cat / Dog / Overall accuracy bars
├── prediction_cat-pet-animal-domestic-104827.png     — confidence chart (Cat 100%)
└── prediction_hero-image1.png                        — confidence chart (Dog 100%)
```

---

## Project Structure

```
L-39-CatsAndDogsNeuralNet/
├── CLAUDE.md                  <- Full task guide for the Claude agent
├── tasks.json                 <- Structured task list (T-001 - T-011)
├── requirements.txt           <- Python dependencies
├── model.py                   <- CNN model definitions (CustomCNN, ResNet-18/50)
├── dataset.py                 <- Oxford Pets data loading + augmentation
├── train.py                   <- GPU training script (main entry point)
├── predict.py                 <- Single-image inference
├── visualize.py               <- Post-training visualization suite
├── CatsAndDogs_Colab.ipynb    <- Google Colab end-to-end notebook
├── data/                      <- Auto-created - Oxford Pets dataset (~750 MB)
└── output/                    <- Auto-created - checkpoints and plots
```

---

## Dataset

**Oxford-IIIT Pet Dataset** — public, free, auto-downloaded by torchvision.

| Property | Value |
|---|---|
| Source | `torchvision.datasets.OxfordIIITPet` |
| Total images | ~3,680 (trainval split) |
| Classes | Cat -> `0`, Dog -> `1` |
| Train / Val split | 80 / 20 (2,944 / 736) |
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

### Install PyTorch with CUDA

```bash
# CUDA 12.4 (RTX 30/40 series):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.8+ / RTX 5000 series (Blackwell sm_120) — requires nightly:
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
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
  --num-workers 0         \   # use 0 on Windows to avoid DataLoader errors
  --output-dir ./output
```

### Model comparison

| Model | Params | Expected Val Acc | GPU time / epoch |
|---|---|---|---|
| `custom_cnn` | ~6 M | ~85-88 % | ~5 min |
| `resnet18` | ~11 M | ~92-95 % | ~2 min |
| `resnet50` | ~25 M | ~95-97 % | ~4 min |

> Achieved **99.73 %** with ResNet-18 on RTX 5080 — well above the expected range due to the powerful GPU enabling stable training with AMP.

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

## Visualizations

Run after training to generate all plots from the saved checkpoint and history:

```bash
python visualize.py
```

Options:

```bash
python visualize.py --checkpoint output/best_model.pth --data-dir ./data --output-dir ./output
```

### Generated plots

| File | Description |
|---|---|
| `training_curves.png` | Basic train/val loss and accuracy curves |
| `training_curves_enhanced.png` | Dark-theme version with best-epoch marker and overfitting gap shading |
| `confusion_matrix.png` | 2x2 TP/TN/FP/FN heatmap with overall accuracy |
| `sample_predictions.png` | 24-image grid — green border = correct, red = wrong |
| `confidence_distribution.png` | P(Dog) histogram split by true class (Cat vs Dog zone) |
| `per_class_accuracy.png` | Bar chart: Cat accuracy, Dog accuracy, Overall |

---

## Google Colab

1. Upload `CatsAndDogs_Colab.ipynb` to Google Drive.
2. Open with Google Colab (right-click -> Open with -> Colab).
3. Set GPU: `Runtime` -> `Change Runtime Type` -> `T4 GPU`.
4. Run all: `Runtime` -> `Run all` (`Ctrl + F9`).

The notebook covers: GPU check -> package install -> data download -> model definition -> training loop -> loss/accuracy plots -> confusion matrix -> sample predictions -> model download.

---

## Architecture

### ResNet-18 (default)

```
ResNet-18 backbone (pretrained ImageNet weights)
  18 residual blocks
  Final FC replaced with:
    Dropout(0.3) -> Linear(512 -> 1)
Output: raw logit -> sigmoid -> P(Dog)
```

### Custom CNN

```
Input: (B, 3, 224, 224)
  ConvBlock( 3 ->  32) + MaxPool -> (B,  32, 112, 112)
  ConvBlock(32 ->  64) + MaxPool -> (B,  64,  56,  56)
  ConvBlock(64 -> 128) + MaxPool -> (B, 128,  28,  28)
  ConvBlock(128-> 256) + MaxPool -> (B, 256,  14,  14)
  AdaptiveAvgPool2d(4,4)         -> (B, 256,   4,   4)
  Flatten -> Linear(4096->512->128->1)
Output: raw logit
```

---

## Training Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Loss | `BCEWithLogitsLoss` | Numerically stable binary CE |
| Optimizer | `AdamW` | Weight decay 1e-4 |
| Scheduler | `CosineAnnealingLR` | Smooth LR decay over all epochs |
| AMP | Enabled (CUDA only) | Mixed precision, ~2x speedup |
| Grad clipping | max_norm = 1.0 | Prevents gradient explosion |
| Augmentation | Flip, Rotate, ColorJitter | Reduces overfitting |
| Input size | 224 x 224 | Standard ImageNet resolution |
| Batch size | 32 | Fits comfortably on 4+ GB VRAM |
| Learning rate | 1e-4 | Conservative for fine-tuning |
| Epochs | 15 | Sufficient for >99% val accuracy |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA out of memory | `--batch-size 16` or `--model custom_cnn` |
| Dataset download fails | `--data-dir C:/Datasets/pets` |
| Windows DataLoader error | `--num-workers 0` |
| RTX 5000 series (Blackwell) not recognized | Install PyTorch nightly cu128 (see Setup above) |
| Unicode error on Windows terminal | Already fixed in this codebase (ASCII-only print statements) |

---

## Completion Checklist

- [x] Environment setup complete
- [x] `torch.cuda.is_available()` returns `True`
- [x] First training run completed (`output/best_model.pth` saved)
- [x] `output/training_curves.png` generated
- [x] Validation accuracy >= 90 % — achieved **99.73 %**
- [x] `output/confusion_matrix.png` generated
- [x] `output/sample_predictions.png` generated
- [x] `output/confidence_distribution.png` generated
- [x] `output/per_class_accuracy.png` generated
- [x] Inference on a custom test image (`predict.py`) — Cat 100.0 % confident
- [ ] Colab notebook runs end-to-end
