# 🧠 Multi-View Contrastive Learning with NT-Xent Loss

This project implements a **self-supervised contrastive learning framework** inspired by SimCLR. To address the challenge of limited training data, it uses **multiple augmented views per image** and a **generalized multi-view NT-Xent loss**, allowing the model to learn rich, invariant representations without labels.

---

## 📚 Overview

SimCLR typically uses two views per image to form positive pairs and learn through contrastive loss. This implementation generalizes the idea by supporting **multiple views per image** and training with a **multi-view cross-entropy (NT-Xent) loss**.

The goal is to:
- Maximize agreement between **augmented views of the same image**
- Minimize similarity to **views of different images**

This setup increases the diversity of positive pairs and improves generalization — particularly useful when training data is limited.

---

## 🔁 Multi-View NT-Xent Loss

The loss operates on embeddings shaped `[B, V, D]`, where:
- `B` = number of images
- `V` = number of views per image
- `D` = embedding dimension

For each embedding:
- All other views of the same image are **positives**
- All views of different images are **negatives**
- Self-similarities are masked out

This setup supports **V ≥ 2** and leverages every possible positive/negative combination per batch, 
making it a strong choice for data-scarce environments.

---

## ⚙️ Training Details

- **Backbone**: ResNet-based encoder
- **Projection head**: 2-layer MLP
- **Views per image**: Configurable (e.g. 2, 4)
- **Loss**: Multi-view NT-Xent (contrastive cross-entropy)

---

## 🔧 Data Augmentation

To simulate data diversity, each image is augmented multiple times using:

- `RandomResizedCrop`
- `ColorJitter` (with configurable strength)
- `GaussianBlur` (optional)
- `RandomHorizontalFlip`
- `ToTensor` + ImageNet normalization


---

## 🧪 Validation & t-SNE Visualization

A small validation split is used **only for monitoring purposes** — it does not affect training.

- Softer augmentations are applied to validation data (e.g. just cropping, flip, no blur).
- Every **50 epochs**, a **t-SNE plot** is generated from validation embeddings to visualize 
- how the model organizes the data in feature space.
- At the **end of training**, a final t-SNE plot is generated using the **original (unaugmented)** data to observe how the model encodes the natural distribution.


---

## 📦 Dependencies

- PyTorch
- torchvision
- PyTorch Lightning
- scikit-learn
- matplotlib, seaborn
- wandb

---
## 💻 Environment Setup

Set up the environment using the provided file:

```bash
conda env create -f env.yml
conda activate simclr-env
```

## 🚀 Usage

Train the model using:
```bash
python train.py --stage=train  --gpus=0  --path exp  --config configs/SIM_CLR.yaml

```

Create the tsne-maps:
```bash
python train.py --stage=test  --gpus=0  --path exp  --config configs/SIM_CLR.yaml

```

## 📈 Logging with Weights & Biases (wandb)

All training metrics, losses, and t-SNE plots are logged using **Weights & Biases**.

- Logs include: training loss, learning rate, and t-SNE embeddings every 50 epochs.

