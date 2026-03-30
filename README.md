# StrokeScope

## An Explainable AI System for Hyperacute Haemorrhagic Stroke Triage on Non-Contrast CT

Multi-radiologist consensus segmentation, lesion volumetry, calibrated uncertainty quantification, and simulated human-in-the-loop routing on the Teknofest 2021 national stroke CT dataset.

> Manuscript in preparation — *Target: Nature Medicine*

---

## Key Results

| Metric                         | Value          | 95% CI        |
| ------------------------------ | -------------- | ------------- |
| AUROC (stroke vs. negative)    | 0.9948         | 0.9839–0.9997 |
| Sensitivity                    | 98.2%          | 96.2–99.6%    |
| Specificity                    | 99.6%          | 98.4–99.6%    |
| Haemorrhagic Dice (mean ± SD)  | 0.924 ± 0.052  | 0.917–0.931   |
| Volumetric MAE                 | 4.0 mL         | —             |
| FNR reduction (HITL, simulated)| 80%            | —             |
| Model parameters               | 1.38 M         | —             |

---

## Repository Structure

```text
strokescope/
├── code/
│   ├── 01_data_pipeline.py          # Download, parse, split, Figure 1
│   ├── 02_model_train_evaluate.py   # StrokeScope training + test evaluation
│   ├── 03_figures.py                # Figures 2–6 from saved results
│   ├── 05_stats_and_gradcam.py      # Bootstrap CIs, Grad-CAM, Figure 7
│   └── 07_manuscript_revision.py   # Final Word document generator
├── figures/                         # Generated PNG figures (1–7)
├── results/                         # JSON metrics, split CSVs, NPY arrays
│   ├── dataset_stats.json
│   ├── model_results.json
│   ├── extended_stats.json
│   ├── train_split.csv
│   ├── val_split.csv
│   ├── test_split.csv
│   └── strokescope_best.pt          # Model weights (5.4 MB)
├── requirements.txt
└── README.md
```

---

## Dataset

**Teknofest 2021 Stroke Dataset** — Turkey Ministry of Health / Health Institutes of Turkey

- 6,650 axial brain NCCT slices (512×512 px)
- 4,427 negative · 1,130 ischaemic · 1,093 haemorrhagic
- Pixel-level segmentation masks for all haemorrhagic slices from **7 independent radiologists**
- STAPLE consensus used as segmentation ground truth
- Publicly available: [acikveri.saglik.gov.tr](https://acikveri.saglik.gov.tr/Home/DataSetDetail/1)
- DOI: [10.5152/eurasianjmed.2022.22096](https://doi.org/10.5152/eurasianjmed.2022.22096)

> **Note:** Ischaemic lesion masks are not included in the released dataset. Segmentation, volumetry, and Grad-CAM analyses are therefore scoped to haemorrhagic stroke only.

---

## Model: StrokeScope

**Architecture:** MobileNetV3-Small encoder (1.38M params, ImageNet pretrained) with three parallel heads:

1. **Classification head** — 3-class (negative / ischaemic / haemorrhagic)
2. **Segmentation head** — 5-stage transposed-convolution decoder
3. **Uncertainty head** — scalar score supervised by inter-annotator disagreement maps

**Combined loss:**

```text
L = λ₁·L_cls + λ₂·L_seg + λ₃·L_unc
```

- `L_cls` = Focal loss (γ=2, α=0.25)
- `L_seg` = BCE + Dice (haemorrhagic slices only)
- `L_unc` = MSE vs. per-pixel inter-radiologist disagreement
- λ₁=1.0, λ₂=0.5, λ₃=0.3

**Training:** Adam (lr=1e-3), cosine annealing, 30 epochs, batch 32, Apple MPS hardware.
**Input:** 224×224 px, brain CT window WL 40 HU / WW 80 HU.

---

## Reproducibility

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and prepare dataset (places files in /tmp/stroke_dataset/)
python code/01_data_pipeline.py

# 3. Train and evaluate StrokeScope
python code/02_model_train_evaluate.py

# 4. Generate Figures 2–6
python code/03_figures.py

# 5. Bootstrap CIs + Grad-CAM + Figure 7
python code/05_stats_and_gradcam.py

# 6. Generate final manuscript Word document
python code/07_manuscript_revision.py
```

Random seed: **42** (fixed in all scripts).
Split files: `results/train_split.csv`, `val_split.csv`, `test_split.csv`.

---

## Limitations

This repository presents a **proof-of-concept evaluation** on a slice-level split.
Key limitations documented in the manuscript:

- Slice-level (not patient-level) data splitting — may produce optimistic estimates
- Ischaemic segmentation masks unavailable — volumetry and explainability limited to haemorrhagic stroke
- HITL evaluation is a simulated retrospective analysis; no prospective reader study performed
- 2D slice-based processing (no 3D inter-slice context)
- Grad-CAM NOS = 0.012 — classification-level attribution, not boundary-precise

External patient-level multi-centre validation is required before any clinical deployment.

---

## Citation

```bibtex
@article{strokescope2026,
  title   = {An Explainable AI System for Hyperacute Haemorrhagic Stroke Triage
             on Non-Contrast CT: Multi-Radiologist Consensus Segmentation,
             Lesion Volumetry, and Simulated Human-in-the-Loop Routing},
  author  = {[Author names]},
  journal = {Nature Medicine},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## License

Code: MIT License.
Dataset: subject to Teknofest 2021 open-data agreement — see dataset DOI above.
