# Membership Inference Attack — TML 2026

**Course:** Trustworthy Machine Learning, 2026  
**Institution:** Universität des Saarlandes / CISPA Helmholtz Center for Information Security  
**Author:** Ashwin Kumar, Harini Raj  
**Leaderboard Team:** team_LXXIX  
**Best Leaderboard Score:** 0.059268 (TPR@5%FPR)

---

## Overview

This repository implements a **Membership Inference Attack (MIA)** against a pretrained ResNet-18 image classifier. The goal is to determine, for each sample in a private dataset, whether it was part of the model's training set.

The attack is based on **RMIA** (Zarifzadeh et al., 2024), using 64 shadow models, 12-augmentation test-time averaging (TTA), and per-class demeaning calibration. The final membership score is a weighted blend of two demeaned RMIA variants found via exhaustive GPU grid search over 10 million weight combinations.

---

## Repository Structure

```
├── MIA_final.ipynb        # Main attack notebook — run this to reproduce best result
├── README.md              # This file
└── shadow_checkpoints/    # Cached shadow model weights and scores (generated on first run)
    ├── shadow_000.pt      # Shadow model weights (64 total)
    ├── ...
    ├── scores_v2_000.npz  # 12-TTA log-odds scores (64 total)
    └── ...
```

> **Note:** Shadow model weights (`.pt` files, ~2.5GB total) and score files are not included due to size constraints. They are automatically trained and cached on first run.

---

## How to Reproduce Best Result

### 1. Prerequisites

```bash
conda create -n mia python=3.11 -y
conda activate mia
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas scikit-learn scipy ipykernel
```

### 2. Download Data

Place the following files in the same directory as `MIA_final.ipynb`:

```bash
curl -L "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt"   -o pub.pt
curl -L "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt"  -o priv.pt
curl -L "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt" -o model.pt
```

### 3. Run the Attack

Open `MIA_final.ipynb` in VS Code or Jupyter and **run all cells in order**.

The notebook will:
1. Load the target model (`model.pt`) and both datasets (`pub.pt`, `priv.pt`)
2. Train 64 shadow ResNet-18 models — **skipped automatically if cached**
3. Compute 12-TTA log-odds scores for all shadow models — **skipped if cached**
4. Compute RMIA variants (mean, median, winsorized mean, p25, p75) with per-class demeaning
5. Apply hardcoded blend weights from GPU grid search
6. Generate `submission.csv` with membership scores in `[0, 1]`

> **First run:** ~4 hours on RTX 4060 Laptop GPU (shadow training + scoring)  
> **Subsequent runs:** ~3 minutes (loads all cached scores)

### 4. Submit

Fill in your API key in Section 7 and run the submission cell.

---

## Method Summary

| Component | Details |
|---|---|
| Target model | ResNet-18, 9-class image classifier |
| Shadow models | 64 × ResNet-18, each trained on a random 50% subset of `pub.pt` |
| Shadow training | SGD, lr=0.1, momentum=0.9, weight decay=5×10⁻⁴, cosine LR, 50 epochs |
| Score function | Log-odds of correct class, averaged over 12 TTA augmentations |
| TTA augmentations | Identity, h-flip, v-flip, spatial rolls (×4), brightness ±, noise, 180° rotation, contrast squeeze |
| Attack | RMIA: target score − reference shadow score (median aggregation) |
| Calibration | Per-class demeaning: subtract per-class median computed on `pub.pt` |
| Final blend | `0.833 × RN(RMIA_med_dm) + 0.167 × RN(RMIA_p25_dm)` |
| Weight search | Exhaustive GPU grid search over 10,077,695 weight combinations (~30s on RTX 4060) |
| Evaluation metric | TPR @ 5% FPR |

---

## Results

| Attack Variant | AUC | TPR@5%FPR |
|---|---|---|
| Random baseline | 0.500 | 0.050 |
| Cross-entropy loss | 0.502 | 0.054 |
| Modified entropy | 0.504 | 0.053 |
| RMIA (median, no calibration) | 0.510 | 0.060 |
| RMIA + per-class demean | 0.514 | 0.061 |
| **Ours (blend + per-class demean)** | **0.514** | **0.063** |
| **Leaderboard (private 30% of priv.pt)** | — | **0.059** |

---

## References

1. Zarifzadeh et al. — *Low-Cost High-Power Membership Inference Attacks* (ICML 2024)  
   https://arxiv.org/pdf/2312.03262

2. Carlini et al. — *Membership Inference Attacks From First Principles* (IEEE S&P 2022)  
   https://arxiv.org/pdf/2112.03570

3. Shokri et al. — *Membership Inference Attacks Against Machine Learning Models* (IEEE S&P 2017)  
   https://arxiv.org/pdf/1610.05820
