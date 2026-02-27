# Cats & Dogs Neural Network Classifier

Binary image classifier that identifies **Cat** or **Dog** in any photo using a custom CNN built from scratch in PyTorch. Runs locally on a CUDA GPU or in Google Colab (free T4 GPU).

---

## Architecture — CatDogCNN

4-block convolutional network trained from scratch on 150×150 colour images.

```
Input: (B, 3, 150, 150)
  Conv(3→32,  k=3, pad=0) + ReLU + MaxPool(2) → (B,  32,  74,  74)
  Conv(32→64, k=3, pad=0) + ReLU + MaxPool(2) → (B,  64,  36,  36)
  Conv(64→128,k=3, pad=0) + ReLU + MaxPool(2) → (B, 128,  17,  17)
  Conv(128→128,k=3,pad=0) + ReLU + MaxPool(2) → (B, 128,   7,   7)
  Flatten                                      → (B, 6272)
  Linear(6272 → 512) + ReLU                   → (B, 512)
  Linear(512  → 2)                             → (B, 2)  ← raw logits

Output: raw logits — CrossEntropyLoss applies log_softmax internally.
        At inference, call torch.softmax(output, dim=1) to get [P(Cat), P(Dog)].

Trainable parameters: 3,453,634
```

---

## Requirement Audit

| ID | Requirement | Status |
|---|---|---|
| R1 | Custom layers only — no pretrained shortcuts | ✅ Pass |
| R2 | Layer structure matches spec | ✅ Pass |
| R3 | Confusion matrix generated after evaluation | ✅ Pass |
| R4 | Loss tracked and plotted per epoch | ✅ Pass |
| R5 | Training accuracy tracked and plotted | ✅ Pass |
| R6 | Validation clearly separated from training | ✅ Pass |
| R7 | User can upload/load image via UI | ✅ Pass |
| R8 | Model predicts Cat or Dog | ✅ Pass |
| R9 | Confidence score displayed per prediction | ✅ Pass |

### Bugs Fixed (audit 2026-02-27)

| Bug | Severity | Fix |
|---|---|---|
| `nn.Softmax` inside model + `CrossEntropyLoss` = double-softmax (incorrect loss) | High | Removed Softmax from `model.py`. `torch.softmax()` applied in `predict.py`, `app.py`, `visualize.py` at inference. |
| `dataset.py` returned `float32` labels; `CrossEntropyLoss` requires `torch.long` | Medium | Changed label dtype in `OxfordPetBinaryDataset.__getitem__`. |
| Confusion matrix only in `visualize.py` (optional) — not in `train.py` | Medium | Added confusion matrix generation at end of `train.py` `main()`. Saves to `output/confusion_matrix.png` automatically. |

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
```

~13 seconds per epoch on RTX 5080 with AMP enabled.

---

## Training Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Loss | `CrossEntropyLoss` | Expects raw logits — do NOT apply Softmax inside model |
| Optimizer | `AdamW` | Weight decay 1e-4 |
| Scheduler | `CosineAnnealingLR` | Smooth LR decay over all epochs |
| AMP | Enabled (CUDA only) | Mixed precision, ~2x speedup |
| Grad clipping | max_norm = 1.0 | Prevents gradient explosion |
| Augmentation | Flip, Rotate, ColorJitter | Reduces overfitting |
| Input size | 150 x 150 | Custom CNN resolution |
| Batch size | 32 | Fits comfortably on 4+ GB VRAM |
| Learning rate | 1e-4 | Default starting rate |
| Epochs | 15 | Recommended for convergence |
| Label dtype | `torch.long` | Required by `CrossEntropyLoss` |

---

## Project Structure

```
L-39-CatsAndDogsNeuralNet/
├── CLAUDE.md                  <- Full task guide for the Claude agent
├── tasks.json                 <- Structured task list with audit results
├── requirements.txt           <- Python dependencies (incl. gradio>=4.0.0)
├── model.py                   <- CatDogCNN — raw logits output, no Softmax
├── dataset.py                 <- Oxford Pets data loading (torch.long labels)
├── train.py                   <- GPU training + auto confusion matrix
├── predict.py                 <- CLI inference (torch.softmax at output)
├── app.py                     <- Gradio web UI (upload + histogram)
├── visualize.py               <- Post-training visualization suite (5 plots)
├── CatsAndDogs_Colab.ipynb    <- Google Colab notebook (needs update — T-009)
├── data/                      <- Auto-created — Oxford Pets dataset (~750 MB)
└── output/                    <- Auto-created — checkpoints and plots
    ├── best_model.pth         <- Best model weights
    ├── training_history.json  <- Per-epoch loss/accuracy log
    ├── training_curves.png    <- Loss & accuracy curves (auto, from train.py)
    └── confusion_matrix.png   <- Confusion matrix (auto, from train.py)
```

---

## Dataset

**Oxford-IIIT Pet Dataset** — public, free, auto-downloaded by torchvision.

| Property | Value |
|---|---|
| Source | `torchvision.datasets.OxfordIIITPet` |
| Total images | ~7,349 (both splits) |
| Cats | ~2,371 |
| Dogs | ~4,978 |
| Default split | 80/20 of trainval (2,944 train / 736 val) |
| Label dtype | `torch.long` — 0=Cat, 1=Dog |
| Download size | ~750 MB |
| License | CC-BY-SA 4.0 |

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
.venv\Scripts\activate        # Windows
source .venv/bin/activate      # Mac / Linux
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
  --source        oxford  \   # oxford (auto-download) | local
  --data-dir      ./data  \
  --epochs        15      \
  --batch-size    32      \
  --lr            0.0001  \
  --num-workers   0       \   # use 0 on Windows to avoid DataLoader errors
  --output-dir    ./output \
  --train-per-class 2000  \   # balanced: 2000 cats + 2000 dogs for training
  --val-samples   2000        # 2000 validation images from remainder
```

### Auto-generated outputs after training

```
output/
├── best_model.pth          <- Best checkpoint (lowest val loss)
├── training_history.json   <- Per-epoch metrics
├── training_curves.png     <- Loss & accuracy line plots
└── confusion_matrix.png    <- 2x2 TP/TN/FP/FN heatmap  ← auto-generated
```

---

## Inference

### CLI

```bash
python predict.py --image path/to/cat.jpg
python predict.py --image path/to/dog.png --chart    # saves confidence chart
python predict.py --upload                           # native file browser dialog
```

Example output:

```
=============================================
  Image  : path/to/cat.jpg
  Result : Cat  (98.0% confident)
  P(Cat) : 98.0%
  P(Dog) :  2.0%
=============================================
[Predict] Chart saved -> output/prediction_cat.png
```

### Web UI

```bash
python app.py
# Opens http://localhost:7860
```

- Drag-and-drop or click to upload any image
- Prediction triggers automatically on upload
- Shows Cat/Dog label probabilities (`gr.Label`)
- Shows confidence histogram bar chart

---

## Visualizations

Run after training for enhanced plots:

```bash
python visualize.py
```

| File | Description |
|---|---|
| `training_curves_enhanced.png` | Dark-theme loss & accuracy with best-epoch marker |
| `confusion_matrix.png` | 2×2 TP/TN/FP/FN heatmap (also auto-generated by `train.py`) |
| `sample_predictions.png` | 3×3 grid (9 images) — green=correct, red=wrong |
| `confidence_distribution.png` | P(Dog) histogram by true class |
| `per_class_accuracy.png` | Cat / Dog / Overall accuracy bars |

---

## Google Colab

1. Upload `CatsAndDogs_Colab.ipynb` to Google Drive.
2. Open with Google Colab (right-click -> Open with -> Colab).
3. Set GPU: `Runtime` -> `Change Runtime Type` -> `T4 GPU`.
4. Run all: `Runtime` -> `Run all` (`Ctrl + F9`).

> **Note (T-009):** The notebook currently references the old ResNet-18 architecture and needs updating to CatDogCNN. See `tasks.json` T-009 for details.

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
- [x] CatDogCNN architecture verified on GPU (RTX 5080, ~13s/epoch)
- [x] Model outputs raw logits — `torch.softmax()` applied at inference
- [x] Dataset labels are `torch.long` — compatible with `CrossEntropyLoss`
- [x] Confusion matrix auto-generated by `train.py` after each training run
- [x] Gradio web UI (`app.py`) — upload + Cat/Dog prediction + histogram
- [x] CLI inference (`predict.py`) — `--chart` saves confidence PNG
- [x] Visualization suite (`visualize.py`) — 5 plots including 3×3 sample grid
- [ ] Full training run (≥ 30 epochs) for validation accuracy > 90%
- [ ] Colab notebook updated for CatDogCNN (T-009)
