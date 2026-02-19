# Claude.md — Complete Task Guide
## Cats & Dogs Neural Network Classifier
### Backend Developer Agent | PyTorch | GPU Training | Google Colab

---

## 🎯 Project Goal

Train a deep learning binary classifier that identifies whether an image contains a **Cat** or a **Dog** using PyTorch. The same model runs both locally on a **CUDA GPU** and in **Google Colab**.

---

## 📁 Project Structure

```
L-39-CatsAndDogsNeuralNet/
├── claude.md                  ← This file (complete guide)
├── tasks.json                 ← All project tasks in JSON format
├── requirements.txt           ← Python dependencies
├── model.py                   ← CNN model definitions
├── dataset.py                 ← Data loading (Oxford Pets auto-download)
├── train.py                   ← GPU training script (main entry point)
├── predict.py                 ← Single-image inference
├── CatsAndDogs_Colab.ipynb    ← Google Colab notebook
├── data/                      ← Auto-created (Oxford Pets ~750 MB)
└── output/                    ← Auto-created (checkpoints, plots)
    ├── best_model.pth         ← Best model weights
    ├── training_history.json  ← Loss/accuracy logs
    └── training_curves.png    ← Training plot
```

---

## 🗂️ Dataset

**Oxford-IIIT Pet Dataset** (public, free, auto-downloaded by torchvision)

| Property | Value |
|---|---|
| Source | `torchvision.datasets.OxfordIIITPet` |
| Total images | ~7,349 |
| Classes | Cat (37 breeds → 0), Dog (37 breeds → 1) |
| Download size | ~750 MB |
| License | CC-BY-SA 4.0 |
| First run | Downloaded automatically to `./data/` |

No manual download needed. The dataset is fetched from Oxford's servers on first run.

---

## ⚙️ Step 1 — Environment Setup

### Option A: Conda Environment (Recommended)
```bash
conda create -n catdog python=3.10 -y
conda activate catdog
pip install -r requirements.txt
```

### Option B: venv
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### Verify GPU
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Install PyTorch with CUDA (if needed)
```bash
# CUDA 12.1 (adjust for your CUDA version):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚀 Step 2 — Train the Model (Local GPU)

### Quick Start (Recommended)
```bash
python train.py
```
This runs **ResNet-18** with Oxford Pets dataset for 15 epochs. Dataset downloads automatically on first run.

### All CLI Options

```bash
python train.py \
  --model      resnet18   \   # custom_cnn | resnet18 | resnet50
  --source     oxford     \   # oxford (auto-download) | local (your data)
  --data-dir   ./data     \   # where to store/read dataset
  --epochs     15         \   # training epochs
  --batch-size 32         \   # images per batch
  --lr         0.0001     \   # initial learning rate
  --num-workers 4         \   # parallel data loading workers
  --output-dir ./output       # where to save checkpoints
```

### Model Comparison

| Model | Params | Expected Val Acc | Training Time (GPU) |
|---|---|---|---|
| `custom_cnn` | ~6M | ~85–88% | ~5 min / epoch |
| `resnet18` | ~11M | ~92–95% | ~2 min / epoch |
| `resnet50` | ~25M | ~95–97% | ~4 min / epoch |

### Expected Output
```
============================
  Cats vs Dogs Neural Network Trainer
============================
[GPU] CUDA device: NVIDIA GeForce RTX 3060
[Dataset] Using Oxford-IIIT Pet Dataset (auto-download to ./data)
[Dataset] Train=5,879  Val=1,470
[Model] Building 'resnet18' ...
[Model] Trainable parameters: 11,178,561

Epoch [  1/15]  Train Loss: 0.4521  Acc: 78.3%  |  Val Loss: 0.3210  Acc: 85.6%
Epoch [  2/15]  Train Loss: 0.2890  Acc: 88.1%  |  Val Loss: 0.2150  Acc: 91.2%  ✓ BEST
...
Epoch [ 15/15]  Train Loss: 0.1043  Acc: 96.2%  |  Val Loss: 0.1821  Acc: 94.7%  ✓ BEST
```

---

## 🔍 Step 3 — Run Inference

```bash
# Predict on any image:
python predict.py --image path/to/cat.jpg

# With custom checkpoint:
python predict.py --image path/to/dog.png --checkpoint output/best_model.pth
```

### Expected Output
```
=============================================
  Image : path/to/cat.jpg
  Result: Cat  (96.3% confident)
  P(Cat): 96.3%
  P(Dog): 3.7%
=============================================
```

---

## ☁️ Step 4 — Google Colab (Cloud GPU)

### Steps
1. **Upload** `CatsAndDogs_Colab.ipynb` to Google Drive
2. **Open** it in Google Colab (right-click → Open with → Colab)
3. **Set GPU Runtime**: `Runtime` → `Change Runtime Type` → `T4 GPU`
4. **Run All**: `Runtime` → `Run all` (or `Ctrl + F9`)

### What each section does

| Cell Section | Description |
|---|---|
| 🔧 GPU Check | Verifies CUDA, prints GPU info |
| 📦 Install | `!pip install torch torchvision tqdm matplotlib` |
| 💾 Data | Auto-downloads Oxford Pets into `/content/data/` |
| 🧠 Model | Defines ResNet-18 (transfer learning) |
| 🏋️ Train | Full training loop with progress bar |
| 📊 Plots | Loss and accuracy curves |
| 🔍 Infer | Predict on sample images from the dataset |
| 💾 Save | Downloads `best_model.pth` to your local machine |

---

## 🧠 Model Architecture

### ResNet-18 (Default, Recommended)
```
ResNet-18 Backbone (pretrained on ImageNet)
  └── 18 layers of residual blocks
  └── FC Head replaced with:
       Dropout(0.3) → Linear(512 → 1)
  Output: raw logit → sigmoid → P(Dog)
```

### Custom CNN
```
Input:  (B, 3, 224, 224)
  ↓ ConvBlock(3  → 32)   MaxPool → (B, 32, 112, 112)
  ↓ ConvBlock(32 → 64)   MaxPool → (B, 64, 56, 56)
  ↓ ConvBlock(64 → 128)  MaxPool → (B, 128, 28, 28)
  ↓ ConvBlock(128→ 256)  MaxPool → (B, 256, 14, 14)
  ↓ AdaptiveAvgPool2d(4,4)       → (B, 256, 4, 4)
  ↓ Flatten + Linear(4096→512→128→1)
  Output: raw logit
```

---

## 📊 Training Details

| Hyperparameter | Value | Notes |
|---|---|---|
| Loss Function | `BCEWithLogitsLoss` | Numerically stable binary CE |
| Optimizer | `AdamW` | Weight decay 1e-4 |
| Scheduler | `CosineAnnealingLR` | Smooth LR decay |
| AMP | Enabled (CUDA only) | Mixed precision for speed |
| Grad Clipping | max_norm=1.0 | Prevents gradient explosion |
| Augmentation | Flip, Rotate, ColorJitter | Reduces overfitting |
| Input Size | 224 × 224 | Standard ImageNet size |

---

## 🐛 Troubleshooting

### CUDA Out of Memory (OOM)
```bash
# Reduce batch size:
python train.py --batch-size 16

# Or use smaller model:
python train.py --model custom_cnn --batch-size 32
```

### Dataset Download Fails
```bash
# Set a different temp directory and retry:
python train.py --data-dir C:/Datasets/pets
```

### Windows num_workers error
```bash
# Use 0 workers on Windows if DataLoader errors:
python train.py --num-workers 0
```

### Slow training (CPU)
```bash
# Ensure CUDA PyTorch is installed:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 📋 Tasks JSON Reference

All project tasks are tracked in [`tasks.json`](./tasks.json). Each task includes:
- Phase, priority, status
- Step-by-step instructions
- Acceptance criteria
- Related files

```bash
# View task summary:
python -c "import json; [print(f'[{t[\"id\"]}] {t[\"title\"]} — {t[\"status\"]}') for t in json.load(open('tasks.json'))['tasks']]"
```

---

## 🧪 Running Tests / Syntax Checks

```bash
# Check syntax of all Python files:
python -m py_compile model.py dataset.py train.py predict.py && echo "All OK"

# Verify notebook JSON:
python -m json.tool CatsAndDogs_Colab.ipynb > nul && echo "Notebook JSON valid"

# Quick model forward pass test:
python model.py

# Quick dataset batch test (downloads Oxford Pets):
python dataset.py
```

---

## ✅ Completion Checklist

- [ ] Environment setup (Step 1)
- [ ] GPU verified with `torch.cuda.is_available()`
- [ ] First training run completed (Step 2)
- [ ] `output/best_model.pth` saved
- [ ] `output/training_curves.png` generated
- [ ] Validation accuracy ≥ 90%
- [ ] Inference on a test image (Step 3)
- [ ] Colab notebook runs end-to-end (Step 4)
