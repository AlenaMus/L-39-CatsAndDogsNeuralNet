# Cats & Dogs Neural Network Classifier

Binary image classifier that identifies **Cat** or **Dog** in any photo using a custom CNN built from scratch in PyTorch. Runs locally on a CUDA GPU or in Google Colab (free T4 GPU).

---

## Architecture — CatDogCNN

4-block convolutional network trained from scratch on 150×150 colour images.

```
Input: (B, 3, 150, 150)
  Conv(3→32,  k=3) + ReLU + MaxPool(2) → (B,  32,  74,  74)
  Conv(32→64, k=3) + ReLU + MaxPool(2) → (B,  64,  36,  36)
  Conv(64→128,k=3) + ReLU + MaxPool(2) → (B, 128,  17,  17)
  Conv(128→128,k=3)+ ReLU + MaxPool(2) → (B, 128,   7,   7)
  Flatten                               → (B, 6272)
  Linear(6272 → 512) + ReLU            → (B, 512)
  Linear(512  → 2)   + Softmax         → (B, 2)

Output: [P(Cat), P(Dog)]   Trainable params: 3,453,634
```

---

## GPU Verification

Architecture confirmed running on NVIDIA GeForce RTX 5080 (17.1 GB VRAM):

```
[GPU] CUDA: NVIDIA GeForce RTX 5080
[GPU] VRAM: 17.1 GB
[Model] Building CatDogCNN (input 150x150) ...
[Model] Trainable parameters: 3,453,634
[Train] AMP: True
Epoch [1/3]  Train Loss: 0.6246  Acc: 67.9%  |  Val Loss: 0.6215  Acc: 67.4%  (12.9s)
Epoch [2/3]  Train Loss: 0.6149  Acc: 67.9%  |  Val Loss: 0.6149  Acc: 67.7%  (12.7s)
Epoch [3/3]  Train Loss: 0.6063  Acc: 68.3%  |  Val Loss: 0.6104  Acc: 68.1%  (12.7s)
```

~13 seconds per epoch on RTX 5080 with AMP enabled.

---

## Training Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Loss | `CrossEntropyLoss` | 2-class output (Cat / Dog) |
| Optimizer | `AdamW` | Weight decay 1e-4 |
| Scheduler | `CosineAnnealingLR` | Smooth LR decay over all epochs |
| AMP | Enabled (CUDA only) | Mixed precision, ~2x speedup |
| Grad clipping | max_norm = 1.0 | Prevents gradient explosion |
| Augmentation | Flip, Rotate, ColorJitter | Reduces overfitting |
| Input size | 150 x 150 | Custom CNN resolution |
| Batch size | 32 | Fits comfortably on 4+ GB VRAM |
| Learning rate | 1e-4 | Default starting rate |
| Epochs | 15 | Recommended for convergence |

---

## Project Structure

```
L-39-CatsAndDogsNeuralNet/
├── CLAUDE.md                  <- Full task guide for the Claude agent
├── tasks.json                 <- Structured task list
├── requirements.txt           <- Python dependencies
├── model.py                   <- CatDogCNN definition + build_model()
├── dataset.py                 <- Oxford Pets data loading + augmentation
├── train.py                   <- GPU training script (main entry point)
├── predict.py                 <- Single-image inference
├── visualize.py               <- Post-training visualization suite
├── CatsAndDogs_Colab.ipynb    <- Google Colab end-to-end notebook
├── data/                      <- Auto-created — Oxford Pets dataset (~750 MB)
└── output/                    <- Auto-created — checkpoints and plots
    ├── best_model.pth         <- Best model weights
    ├── training_history.json  <- Per-epoch loss/accuracy log
    └── training_curves.png    <- Loss & accuracy curves
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
  --source     oxford     \   # oxford (auto-download) | local (your own data)
  --data-dir   ./data     \
  --epochs     15         \
  --batch-size 32         \
  --lr         0.0001     \
  --num-workers 0         \   # use 0 on Windows to avoid DataLoader errors
  --output-dir ./output
```

### Expected output

```
============================================================
  Cats vs Dogs Neural Network Trainer
  Backend Developer Agent | PyTorch
============================================================
[GPU] CUDA: NVIDIA GeForce RTX 5080
[GPU] VRAM: 17.1 GB
[Dataset] Source: Oxford-IIIT Pet (auto-download to ./data)
[Dataset] Train: 2,944  |  Val: 736

[Model] Building CatDogCNN (input 150x150) ...
[Model] Trainable parameters: 3,453,634

[Train] Epochs: 15  |  Batch: 32  |  AMP: True
[Train] Checkpoints -> output
------------------------------------------------------------
Epoch [  1/15]  Train Loss: 0.62  Acc: 68.0%  |  Val Loss: 0.62  Acc: 67.4%  ( 13s)
Epoch [  2/15]  Train Loss: 0.59  Acc: 70.5%  |  Val Loss: 0.59  Acc: 71.1%  * BEST
...
Epoch [ 15/15]  Train Loss: 0.40  Acc: 83.0%  |  Val Loss: 0.42  Acc: 82.5%  * BEST
```

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

Run after training to generate all plots from the saved checkpoint:

```bash
python visualize.py
python visualize.py --checkpoint output/best_model.pth --data-dir ./data --output-dir ./output
```

### Generated plots

| File | Description |
|---|---|
| `training_curves.png` | Train/val loss and accuracy curves |
| `training_curves_enhanced.png` | Dark-theme version with best-epoch marker |
| `confusion_matrix.png` | 2×2 TP/TN/FP/FN heatmap with overall accuracy |
| `sample_predictions.png` | 24-image grid — green border = correct, red = wrong |
| `confidence_distribution.png` | P(Dog) histogram split by true class |
| `per_class_accuracy.png` | Bar chart: Cat accuracy, Dog accuracy, Overall |

---

## Google Colab

1. Upload `CatsAndDogs_Colab.ipynb` to Google Drive.
2. Open with Google Colab (right-click -> Open with -> Colab).
3. Set GPU: `Runtime` -> `Change Runtime Type` -> `T4 GPU`.
4. Run all: `Runtime` -> `Run all` (`Ctrl + F9`).

The notebook covers: GPU check -> package install -> data download -> model definition -> training loop -> loss/accuracy plots -> confusion matrix -> sample predictions -> model download.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA out of memory | `--batch-size 16` |
| Dataset download fails | `--data-dir C:/Datasets/pets` |
| Windows DataLoader error | `--num-workers 0` |
| RTX 5000 series (Blackwell) not recognized | Install PyTorch nightly cu128 (see Setup above) |
| Unicode error on Windows terminal | Already fixed — ASCII-only print statements |

---

## Completion Checklist

- [x] Environment setup complete
- [x] `torch.cuda.is_available()` returns `True`
- [x] CatDogCNN architecture verified on GPU (RTX 5080)
- [ ] Full training run completed (`output/best_model.pth` saved)
- [ ] `output/training_curves.png` generated
- [ ] Validation accuracy >= 80 %
- [ ] Visualizations generated (`visualize.py`)
- [ ] Inference on a custom test image (`predict.py`)
- [ ] Colab notebook runs end-to-end
