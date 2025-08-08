# UQ-SegTorch — Uncertainty Estimation for Semantic Segmentation (PyTorch)

A **PyTorch framework** for **uncertainty estimation** on semantic segmentation. It implements **Vanilla U-Net** and **Iterative U-Net (DRU)**, supports **MC Dropout**, **Ensembles (via repeated checkpoints/runs)**, **iterative refinement**, and provides **CLI tools** for training, prediction, and evaluation with **segmentation** and **calibration** metrics.  
**Multi-GPU** execution (DDP) is supported for both training and inference.

---

## ⚙️ Environment Setup

```bash
git clone https://github.com/your-username/uq-segtorch.git
cd uq-segtorch

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> Tested with Python 3.9–3.11, PyTorch ≥ 2.2, CUDA 11.8/12.x.

---

## 📦 Project Overview

- **Models**
  - `models/unet.py`: Vanilla U-Net (supports `norm_type`: batch/group/none; dropout).
  - `models/druv2.py`: Iterative U-Net with **ConvDRU** recurrent block (`druv2`, `dru`), iterative steps, feedback channel.
- **Datasets**
  - Factory in `custom_dataset/dataset_factory.py` exposes: `RT` (RoadTracer), `Mass` (Massachusetts), `Drive` (DRIVE), `Nuclei` (Cell nuclei, if implemented in your DatasetController).
- **DDP Utilities**
  - `ddp_setup.py`: `ddp_setup`, `ddp_cleanup`, `find_free_port`.
- **Losses & Metrics**
  - `custom_loss.py`: BCE, Dice, Dice+BCE, Dice+Focal, IoU, Focal.
  - `metrics.py`: CCQ (normal/relaxed), AULC, rAULC, correlation, crop-based stats, pixel-wise STD uncertainty.

---

## 🏗️ Model Highlights

### Vanilla U-Net (PyTorch)
- Encoder–decoder with skip connections, transposed convolutions for upsampling.
- Configurable `dropout_rate` and `norm_type` (`batch`, `group`, `none`).
- Output: single-channel logit map (use Sigmoid for probabilities).

### Iterative U-Net — DRU / DRUv2
- U-Net backbone + **ConvDRU** recurrent unit at bottleneck; runs for `steps` iterations.
- **Feedback channel**: input is concatenated with previous prediction `s` each step.
- `druv2` adds options like `feature_scale`, `hidden_size`, `dropout_rate` in early blocks.
- Returns list of step outputs (and hidden states for `druv2`).

---

## 🧪 Training (DDP-ready)

```bash
python train.py \\
  --dataset RT \\
  --dataset_path /data/roadtracer \\
  --num_workers 16 \\
  --model iterative \\
  --dropout_rate 0.3 \\
  --norm_type group \\
  --image_channel 3 \\
  --add_channel \\
  --batch_size 4 \\
  --learning_rate 1e-3 \\
  --loss_function dice_focal \\
  --num_epoch 100 \\
  --save_path ./checkpoints/iter_dru_rt \\
  --save_per_epoch 5 \\
  --gpus 0,1
```

**What happens:** `train.py` spawns **one process per GPU** (DDP). Global batch size = `batch_size × world_size`. Config is saved to `<save_path>/setting.yaml`, logs to `<save_path>/train_logs/…`.

### Training Arguments

| Arg | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `--dataset` | str | ✔ | – | One of: `RT`, `Mass`, `Drive`, `Nuclei` (if implemented). |
| `--dataset_path` | str | ✔ | – | Root path to dataset. |
| `--num_workers` | int | ✖ | 32 | PyTorch DataLoader workers per process. |
| `--model` | str | ✔ | – | `iterative` (DRU/feedback) or `vanila` (Vanilla U-Net). |
| `--dropout_rate` | float | ✖ | 0.1 | Dropout in conv blocks. |
| `--norm_type` | str | ✔ | `None` | One of: `batch`, `group`, `instance`, `none`. |
| `--image_channel` | int | ✖ | 3 | Input channels of raw image. |
| `--add_channel` | flag | ✖ | False | Adds 1 feedback channel to inputs (needed for iterative). |
| `--batch_size` | int | ✔ | – | Per-GPU batch size. |
| `--learning_rate` | float | ✖ | 0.001 | Optimizer lr. |
| `--loss_function` | str | ✔ | `focal` | `focal`, `iou`, `bce`, `dice`, `dice_bce`, `dice_focal`. |
| `--num_epoch` | int | ✔ | – | Total epochs. |
| `--save_path` | str | ✔ | – | Checkpoints + logs dir. |
| `--save_per_epoch` | int | ✖ | 5 | Save every N epochs. |
| `--gpus` | str | ✔ | – | Comma-separated GPU IDs (e.g., `0,1,2`). |

> **Tip:** For iterative models, effective input channels = `image_channel + 1` when `--add_channel` is set.

---

## 🔮 Prediction (DDP-ready)

```bash
python predict.py \\
  --dataset RT \\
  --dataset_path /data/roadtracer \\
  --model_path ./checkpoints/iter_dru_rt \\
  --epoch 95 \\
  --save_path ./preds/iter_dru_rt \\
  --training_mode true \\
  --batch_size 1 \\
  --samples 8 \\
  --gpus 0,1
```

- **`--training_mode true`**: Enables dropout during inference (MC Dropout).
- **`--samples` > 1** required if training mode is true; must be `1` if false.
- Predictions are saved in `<save_path>/epoch_<epoch>/…`.

---

## 📊 Evaluation

```bash
python evaluate.py \\
  --metric_type calibration \\
  --prediction_path ./preds/iter_dru_rt \\
  --epoch 95 \\
  --model_type iterative \\
  --samples 8 \\
  --relaxed_ccq true \\
  --n_rows 2 \\
  --n_cols 2 \\
  --save_path ./eval_results \\
  --bad_sample_path ./bad_samples
```

**Metric Types**
- `segmentation`: CCQ, F1, IoU.
- `calibration`: AULC, rAULC, corr, ECE-like metrics.
- `out-of-distribution`: Not yet implemented.

---

## 📜 License

MIT License
