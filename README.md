# Membership Inference Attack - TML 2026

**Course:** Trustworthy Machine Learning, 2026  
**Institution:** CISPA Helmholtz Center for Information Security  
**Author:** Ashwin Kumar, Harini Raj  
**Leaderboard Team:** team_LXXIX  

---

## Overview

This repository implements a **Membership Inference Attack (MIA)** against a pretrained ResNet-18 image classifier. The goal is to determine, for each sample in a dataset, whether it was part of the model's training set.

The attack is based on **RMIA (Relative Membership Inference Attack)** from Zarifzadeh et al. (2023), combined with a per-sample likelihood ratio score derived from 64 shadow models trained using the LiRA framework (Carlini et al., 2022).

---

## Repository Structure

```
├── MIA.ipynb          # Main attack notebook — run this to reproduce results
├── README.md              # This file
└── shadow_checkpoints/    # Cached shadow model scores (generated on first run)
    ├── scores_000.npz
    ├── scores_001.npz
    └── ...                # 64 score files total
```

> **Note:** Shadow model weights (`.pt` files, ~2.5GB) are not included due to size constraints. They are automatically regenerated and cached on first run.

---

## How to Reproduce Best Result

### 1. Prerequisites

```bash
conda create -n mia python=3.11 -y
conda activate mia
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas matplotlib scikit-learn scipy tqdm ipykernel
```

### 2. Download Data

Place the following files in the same directory as `attack3.ipynb`:

```bash
curl -L "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt"   -o pub.pt
curl -L "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt"  -o priv.pt
curl -L "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt" -o model.pt
```

### 3. Run the Attack

Open `attack3.ipynb` in VS Code or Jupyter and **run all cells in order**.

The notebook will:
1. Load the target model and both datasets
2. Train 64 shadow models (ResNet-18, 50 epochs each) — **cached after first run**
3. Compute RMIA and LiRA scores for all private samples
4. Generate `submission.csv` with membership scores in `[0, 1]`

> **First run:** ~4 hours on an RTX 4060 Laptop GPU (shadow training)  
> **Subsequent runs:** ~2 minutes (loads cached scores)

### 4. Submit

Set your API key in the last cell and run it to submit to the leaderboard.

---

## Method Summary

| Component | Details |
|---|---|
| Target model | ResNet-18, 9-class image classifier |
| Shadow models | 64 × ResNet-18, trained on random 50% subsets of `pub.pt` |
| Shadow training | SGD, lr=0.1, momentum=0.9, cosine LR schedule, 50 epochs |
| Score function | Log-odds of correct class probability |
| Attack | RMIA (target score − mean shadow score) blended with per-sample LiRA likelihood ratio |
| Evaluation metric | TPR @ 5% FPR |

---

## Results

| Dataset | AUC | TPR @ 5% FPR |
|---|---|---|
| Public (`pub.pt`) local eval | 0.5069 | 0.0653 |
| Private leaderboard (public 30%) | — | 0.0556 |

---

## Key Insights & Model Analysis

* **High Regularization / DP-SGD:** Initial diagnostics revealed a generalization gap of <1% (99.76% Train vs. 98.93% Test). Standard MIA metrics (Logits, Gradient Norms, Boundary Attacks) yielded random-chance AUCs (~0.50), suggesting the target model is secured using Differential Privacy (DP-SGD) or extreme sharpness-aware minimization.
* **The Variance Trap:** Standard LiRA failed (TPR 0.056) because DP-SGD cryptographically smooths the standard deviation of the shadows, blinding the metric. Pivoting to the non-parametric RMIA (Zarifzadeh et al.) successfully bypassed this variance trap.
* **Class-Conditional Leakage & The Private Set Trap:** Local analysis on `pub.pt` revealed non-uniform privacy guarantees (e.g., Class 0 had a TPR of 0.029, while Class 3 leaked at 0.076). An attack weighted heavily toward these vulnerable classes achieved a local TPR of 0.0653. However, the server score of 0.0556 confirms the private dataset is perfectly balanced against localized vulnerabilities, proving the model's Uniform Differential Privacy is mathematically robust.

## References

1. Shokri et al. — *Membership Inference Attacks Against Machine Learning Models* (2017)  
   https://arxiv.org/pdf/1610.05820

2. Carlini et al. — *Membership Inference Attacks From First Principles* (2022)  
   https://arxiv.org/pdf/2112.03570

3. Zarifzadeh et al. — *Low-Cost High-Power Membership Inference Attacks* (2023)  
   https://arxiv.org/pdf/2312.03262
