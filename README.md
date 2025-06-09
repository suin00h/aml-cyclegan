# 🌀 CycleGAN Implementation

This repository contains an implementation of CycleGAN for unpaired image-to-image translation. It includes full training and testing pipelines with logging via [Weights & Biases (wandb)].

---

## 📦 Project Structure

```
.
├── configs/              # YAML config files for each dataset (map.yaml, horse2zebra.yaml, etc.)
├── dataset.py            # Dataset loading logic
├── models/               # Generator, Discriminator, CycleGAN wrapper
├── train.py              # Training script
├── test.py               # Testing/inference script
├── evaluate/             # Metrics evaluation scripts (FID, IS, SSIM, LPIPS)
├── results/              # Inference results
├── checkpoints/          # Saved models
└── utils/                # Utility functions (weight loading, buffer, etc.)
```

---

## ⚙️ Requirements

- Python 3.11
- torch 2.7.0
- torchvision 0.22.0
- wandb
- scikit-image
- lpips
- pytorch-fid

---

## 🚀 Training

### 📁 Step 1: Prepare a config YAML file

### ▶️ Step 2: Run training

```bash
python train.py --name map
```

Model checkpoints will be saved every `save_epochs` (e.g., every 10 epochs) in `./checkpoints/map/`.

---

## 🧪 Testing / Inference

### ▶️ Step 1: Run inference with trained models

```bash
python test.py --name map --epoch 200
```

Inference results will be saved to:

```
results/
└── map/
    └── epoch_200/
        ├── real_x/
        ├── fake_y/
        ├── real_y/
        └── fake_x/
```

---

## 📊 Evaluation

Evaluation metrics include:

- **FID**
- **Inception Score**
- **SSIM**
- **PSNR**
- **LPIPS**

```bash
python evaluate/evaluate.py --name map --epoch 200 --size 256 256
```
---

## 📌 Notes

- 🐇 `wandb` is used for real-time training monitoring. To disable, remove `init_wandb()` calls in `train.py`.
- 🧪 Test images are translated both ways: `X → Y` and `Y → X`.
- ✅ Make sure your dataset is structured as:
  ```
  datasets/
  └── maps/
      ├── trainA/
      ├── trainB/
      ├── testA/
      └── testB/
  ```

---

## 📮 Citation

If you use this codebase, consider citing the original CycleGAN paper:

> Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
