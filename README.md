# Rare Cell-Type Identification Under Batch Shift

**Bayesian Active Learning on a LoRA-Adapted Vision Foundation Model**

This repository contains the complete experimental pipeline for a study on identifying rare cell types in microscopy images under extreme class imbalance, annotation scarcity, and covariate shift across imaging batches. The pipeline couples a frozen, self-supervised vision foundation model (**DINOv2**) with **Low-Rank Adaptation (LoRA)** fine-tuning, a **Monte Carlo (MC)-Dropout Bayesian uncertainty head**, and a **class-balanced BALD acquisition function** inside an active-learning (AL) loop.

📄 Paper: *Rare Cell-Type Identification Under Batch Shift: Bayesian Active Learning on a LoRA-Adapted Vision Foundation Model* — Sadia Ruhama, Department of Computer Science and Engineering, BRAC University.

---

## Overview

Rare cell types — drug-resistant cancer cells, rare immune-cell subsets, early-stage disease cells — are visually subtle and appear at extremely low frequency in microscopy data. Standard supervised classifiers struggle because there are too few labeled examples to train on, and models trained on imbalanced data tend to be overconfident on the classes they rarely see.

This project investigates whether a **frozen self-supervised backbone**, adapted with a **parameter-efficient fine-tuning method**, and coupled to a **class-imbalance-aware acquisition function**, can make active learning practical under extreme rarity and real batch-level domain shift.

### Pipeline

```text
Microscopy images → Preprocess & Augment → Frozen DINOv2 ViT Backbone (+ LoRA)
→ Bayesian (MC-Dropout) Uncertainty Head → Class-Balanced BALD Acquisition
→ Active-Learning Loop → Evaluation

| Component | Configuration |
|---|---|
| Backbone | `facebook/dinov2-small` (frozen) |
| Fine-tuning | LoRA on attention Q/V projections (r=8, α=16, dropout=0.05) |
| Trainable parameters | ≈313K / 22.37M total (≈1.4%) |
| Uncertainty head | 2-layer MLP (hidden=256, GELU), always-on dropout (p=0.3) |
| Uncertainty method | MC-Dropout, 20 stochastic forward passes |
| Loss | Class-balanced focal loss (γ=2.0, effective-number reweighting) |
| Acquisition | Class-balanced BALD (inverse-class-frequency-weighted mutual information) |
| AL schedule | 8 rounds × 20 acquired images/round, 80-image seed set |
| Calibration | Post-hoc temperature scaling fit on validation set |


| Notebook | Phase | Purpose |
|---|---|---|
| `Synthetic rare-cell benchmark 1.ipynb` / `2.ipynb` | 1–2 | Procedurally rendered synthetic rare-cell benchmark; confirms the AL loop runs end-to-end under deliberate class overlap |
| `Real-Data Integration with LIVECell.ipynb` | 3 | Integrates real LIVECell microscopy data; constructs the rare-class scenario (BV2, 15 crops) atop a naturally balanced dataset |
| `First real-data AL run .ipynb` | 4 (v1) | First real-data active-learning run; surfaces overfitting (test ECE rises from 0.02 → 0.25 over 8 rounds) |
| `correction.ipynb` | 4 (v2) | Overfitting correction — early stopping, increased weight decay, temperature scaling |
| `Cross-batch split.ipynb` | 5 | Constructs a well-held-out cross-batch split for genuine same-platform generalization testing |
| `Data-quality refinement.ipynb` | 6 | Identifies and fixes a black-padding crop artifact via aspect-preserving, reflection-padded cropping |
| `Acquisition-Function Refinement.ipynb` | 7 | Proposes and benchmarks a full-probability reweighting of the class-balanced BALD acquisition score |
| `Consolidated report v1 .ipynb` | 8 | Aggregates experimental logs into a unified comparison table |
| `Multi-seed validatio.ipynb` | 9 | Multi-seed (n=3) validation diagnosing acquisition-function equivalence under confidence saturation |
| `Linear-probe vs. LoRA.ipynb` | 10 | Fine-tuning-depth ablation comparing a frozen linear probe against LoRA adaptation |
| `Consolidated report v2.ipynb` | 11 | Final consolidated report integrating multi-seed and ablation results |


| Configuration | Macro-F1 | Rare-class F1 | Rare-class AUPRC | ECE (raw → calibrated) |
|---|---|---|---|---|
| Synthetic benchmark | 0.616 | 0.000 | 0.038 | n/a |
| Real, overfit (Phase 4 v1) | 0.504 | 0.224 | 0.468 | 0.254 → — |
| Real, corrected (Phase 4 v2) | 0.563 | 0.376 | 0.596 | 0.130 → 0.055 |
| Same-batch (fixed crops) | 0.590 | 0.274 | 0.519 | 0.099 → — |
| Cross-batch (held-out wells) | 0.629 | 0.214 | 0.627 | 0.084 → — |


## Requirements

```bash
pip install transformers peft accelerate torchao scikit-learn matplotlib timm pycocotools requests tqdm
