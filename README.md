# NC-GAT: Noise-Consistency Regularized Inter-Lead Graph Attention for 12-Lead ECG Classification

Multi-label classification of 6 rhythm/conduction classes (NSR, SB, STach, AF, IAVB, LBBB) from 12-lead ECG, modelling leads as graph nodes with attention and training with a noise-consistency regularizer. Evaluated for robustness (noise, missing leads), cross-dataset generalization, and explanation stability.

**Data:** PTB-XL (train/val/internal test, patient-level split, 0 patient overlap) · Chapman-Shaoxing (external test)
**Stack:** Python, PyTorch, scikit-learn, wfdb

> Research project, not a medical device. Not validated for clinical use.

## TL;DR
- **Model:** per-lead residual CNN encoder → 3 graph-attention layers across leads → temporal attention pooling → classifier. Trained with focal loss, label smoothing, weighted sampling, and a consistency loss between clean and noisy predictions.
- **Internal test (PTB-XL):** NC-GAT macro-F1 0.789 (95% CI 0.766-0.810), AUROC 0.980, beating CNN (0.740) and Transformer (0.609) baselines.
- **Robustness:** under muscle noise (SNR 0 dB) macro-F1 is 0.766 vs 0.294 for a GAT trained without augmentation; with missing limb leads 0.766 vs 0.423.
- **Cross-dataset (PTB-XL → Chapman):** macro-F1 0.631 and AUROC 0.937, the best of all models, but thresholds tuned on PTB-XL transfer poorly (see Limitations).
- **Explanation stability:** attributions are more stable under noise (Spearman 0.81 vs 0.74), but less stable across datasets (0.09 vs 0.46).

## Results

Internal test, PTB-XL (single seed, thresholds tuned per class on validation):

| Model | Macro-F1 | 95% CI (bootstrap) | AUROC | AUPRC |
|---|---|---|---|---|
| CNN 1D | 0.740 | 0.717-0.761 | 0.968 | 0.779 |
| Transformer | 0.609 | 0.585-0.631 | 0.906 | 0.626 |
| GAT | 0.775 | 0.750-0.796 | 0.978 | 0.816 |
| **NC-GAT** | **0.789** | 0.766-0.810 | **0.980** | **0.829** |

McNemar test GAT vs NC-GAT: p = 0.042 (small effect on clean data; CIs overlap).

Robustness (macro-F1, GAT without augmentation → NC-GAT):

| Condition | GAT | NC-GAT |
|---|---|---|
| Clean | 0.775 | 0.789 |
| Baseline wander, SNR 5 dB | 0.509 | 0.783 |
| Muscle noise, SNR 0 dB | 0.294 | 0.766 |
| Missing V1 | 0.744 | 0.781 |
| Missing limb leads | 0.423 | 0.766 |
| Missing precordial leads | 0.324 | 0.733 |

Cross-dataset (macro-F1, PTB-XL → Chapman): CNN 0.592 · Transformer 0.506 · GAT 0.599 · **NC-GAT 0.631**.

## Key figures
![Robustness](results (1)/nc_gat_final/results/eksperimen_c_nc.png)
![Cross-dataset](results (1)/nc_gat_final/results/eksperimen_d_cross.png)
![Attention](results (1)/nc_gat_final/results/attention_heatmap.png)

## Method
1. Label harmonization SCP-ECG → SNOMED-CT, validated against the PhysioNet 2021 reference (5 of 6 classes match exactly; NSR differs by ~3%).
2. Preprocessing: resample to 500 Hz, 0.5-45 Hz Butterworth band-pass, per-lead normalization, pad/crop to 10 s.
3. Patient-level split (GroupShuffleSplit) to prevent leakage.
4. Imbalance handling: weighted sampler + softened pos_weight + focal loss.
5. Noise simulation: baseline wander, powerline, muscle noise at a target SNR, plus missing-lead scenarios.
6. Evaluation: macro/micro F1, AUROC, AUPRC, bootstrap CI, McNemar test, Integrated Gradients stability (Spearman, cosine, top-3 overlap, JS divergence).

## Limitations
- **Single training seed:** CIs reflect test sampling only, not training variance.
- **Robustness comparison is confounded:** NC-GAT is trained with noise/missing-lead augmentation, the GAT baseline is not. An "augmentation without consistency loss" ablation is needed to isolate the effect of the consistency term.
- **Cross-dataset thresholds:** per-class thresholds tuned on PTB-XL transfer poorly (e.g. SB recall 0.12, NSR precision 0.32 on Chapman) even though AUROC stays high; per-domain calibration is future work.
- **Explanation stability across datasets decreases** for NC-GAT.
- Minor NSR label mismatch (~3%) vs the reference mapping.

## Reproduce
```bash
git clone https://github.com/vaurollaaa/ncgat-ecg-classification.git
cd ncgat-ecg-classification
pip install -r requirements.txt
# download the datasets (see data/README.md), set PTBXL_ROOT and CHAPMAN_ROOT
jupyter notebook notebooks/ncgat_ecg_classification.ipynb
```
