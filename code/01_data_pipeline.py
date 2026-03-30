"""
StrokeScope - Data Pipeline
Teknofest 2021 Stroke Dataset
Real data loading, preprocessing, and dataset statistics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/tmp/stroke_dataset")
OUT_DIR = Path("/Users/nusretonen/stroke-paper/results")
FIG_DIR = Path("/Users/nusretonen/stroke-paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ISKEMI_PNG  = BASE / "kanama3" / "Kanama Veri Seti" / "PNG"       # kanama=hemorrhage
ISKEMI_MASK = BASE / "kanama3" / "Kanama Veri Seti" / "OVERLAY"
NEG_PNG     = BASE / "inmeyok3" / "İnme Yok Veri Set_PNG"
CHRONIC_PNG = BASE / "iskemi3" / "İnme Yok Veri Set_PNG"          # chronic/other negatives

# Competition test set
COMP_DIR    = BASE / "competition_1" / "Test Veri Seti_1"
COMP_DICOM  = COMP_DIR / "DICOM"
COMP_LABELS = COMP_DIR / "ASAMA1_Cevaplar.txt"

print("=" * 60)
print("StrokeScope - Data Pipeline")
print("=" * 60)

# ─── 1. Load metadata from Excel ─────────────────────────────────────────────
import openpyxl
wb = openpyxl.load_workbook("/tmp/stroke_info.xlsx")
ws = wb["Training Data Bilgi"]

records = []
for row in ws.iter_rows(min_row=3, values_only=True):
    if row[0] is not None and isinstance(row[0], (int, float)):
        records.append({
            "image_id": int(row[0]),
            "folder": str(row[1]),
            "stroke": int(row[2]),
            "stroke_type": str(row[3]) if row[3] else "N/A"
        })

df = pd.DataFrame(records)
print(f"\n[1] Dataset Overview")
print(f"    Total slices: {len(df):,}")
print(f"    Negative (no stroke): {(df.stroke==0).sum():,}")
print(f"    Positive (stroke):    {(df.stroke==1).sum():,}")
print(f"      - Ischemic:  {(df.stroke_type=='İSKEMİ').sum():,}")
print(f"      - Hemorrhagic: {(df.stroke_type=='KANAMA').sum():,}")

# ─── 2. Image Statistics ─────────────────────────────────────────────────────
print("\n[2] Computing image statistics...")

def analyze_images(img_dir, label, max_samples=200):
    """Compute pixel statistics from CT slices."""
    paths = list(Path(img_dir).glob("*.png"))[:max_samples]
    stats = []
    for p in paths:
        try:
            img = np.array(Image.open(p).convert('L'))  # grayscale
            stats.append({
                'mean': float(img.mean()),
                'std': float(img.std()),
                'min': float(img.min()),
                'max': float(img.max()),
                'nonzero_pct': float((img > 10).mean() * 100)
            })
        except Exception:
            continue
    return pd.DataFrame(stats).assign(label=label)

stats_neg = analyze_images(NEG_PNG, 'Negative')
stats_hem = analyze_images(ISKEMI_PNG, 'Hemorrhagic')
stats_chr = analyze_images(CHRONIC_PNG, 'Chronic/Other')

all_stats = pd.concat([stats_neg, stats_hem, stats_chr], ignore_index=True)
print(f"    Analyzed {len(all_stats)} images")

# ─── 3. Mask Analysis (Segmentation Ground Truth) ─────────────────────────────
print("\n[3] Analyzing segmentation masks...")

mask_paths = sorted(ISKEMI_MASK.glob("*.png"))
mask_stats = []

for mp in mask_paths[:200]:
    orig_path = ISKEMI_PNG / mp.name
    if not orig_path.exists():
        continue
    try:
        mask = np.array(Image.open(mp).convert('L'))
        orig = np.array(Image.open(orig_path).convert('L'))

        # Binary mask: any non-black pixel
        bin_mask = (mask > 10).astype(np.float32)
        lesion_px = bin_mask.sum()
        total_px = bin_mask.size

        # Approximate volume (assuming 5mm slice thickness, 0.5x0.5mm pixel spacing)
        voxel_vol_ml = 0.5 * 0.5 * 5.0 / 1000.0  # mm^3 -> mL
        vol_ml = lesion_px * voxel_vol_ml

        mask_stats.append({
            'image_id': mp.stem,
            'lesion_pixels': int(lesion_px),
            'lesion_pct': float(lesion_px / total_px * 100),
            'volume_ml': float(vol_ml),
            'has_lesion': lesion_px > 100
        })
    except Exception:
        continue

mask_df = pd.DataFrame(mask_stats)
print(f"    Masks analyzed: {len(mask_df)}")
print(f"    With visible lesion: {mask_df.has_lesion.sum()}")
print(f"    Mean lesion volume: {mask_df[mask_df.has_lesion]['volume_ml'].mean():.1f} mL")
print(f"    Median lesion volume: {mask_df[mask_df.has_lesion]['volume_ml'].median():.1f} mL")
print(f"    Volume range: {mask_df[mask_df.has_lesion]['volume_ml'].min():.1f} – {mask_df[mask_df.has_lesion]['volume_ml'].max():.1f} mL")

# ─── 4. Dataset Split ─────────────────────────────────────────────────────────
print("\n[4] Creating train/val/test splits...")

# Use the actual dataset metadata
df_stroke = df[df.stroke == 1].copy()
df_neg    = df[df.stroke == 0].copy()

# Stratified split by stroke type
from sklearn.model_selection import train_test_split

df_stroke_train, df_stroke_temp = train_test_split(
    df_stroke, test_size=0.3, random_state=42, stratify=df_stroke['stroke_type']
)
df_stroke_val, df_stroke_test = train_test_split(
    df_stroke_temp, test_size=0.667, random_state=42, stratify=df_stroke_temp['stroke_type']
)

df_neg_train, df_neg_temp = train_test_split(df_neg, test_size=0.3, random_state=42)
df_neg_val, df_neg_test = train_test_split(df_neg_temp, test_size=0.667, random_state=42)

train_df = pd.concat([df_stroke_train, df_neg_train])
val_df   = pd.concat([df_stroke_val, df_neg_val])
test_df  = pd.concat([df_stroke_test, df_neg_test])

split_summary = {
    'train': {'total': len(train_df), 'positive': train_df.stroke.sum(), 'negative': (train_df.stroke==0).sum()},
    'val':   {'total': len(val_df),   'positive': val_df.stroke.sum(),   'negative': (val_df.stroke==0).sum()},
    'test':  {'total': len(test_df),  'positive': test_df.stroke.sum(),  'negative': (test_df.stroke==0).sum()},
}
for split, s in split_summary.items():
    print(f"    {split:5s}: {s['total']:,} total | {s['positive']:,} positive | {s['negative']:,} negative")

# Save splits
train_df.to_csv(OUT_DIR / "train_split.csv", index=False)
val_df.to_csv(OUT_DIR / "val_split.csv", index=False)
test_df.to_csv(OUT_DIR / "test_split.csv", index=False)

# ─── 5. Save Dataset Statistics ───────────────────────────────────────────────
dataset_stats = {
    'total_slices': len(df),
    'n_negative': int((df.stroke==0).sum()),
    'n_positive': int((df.stroke==1).sum()),
    'n_ischemic': int((df.stroke_type=='İSKEMİ').sum()),
    'n_hemorrhagic': int((df.stroke_type=='KANAMA').sum()),
    'mean_lesion_volume_ml': float(mask_df[mask_df.has_lesion]['volume_ml'].mean()),
    'median_lesion_volume_ml': float(mask_df[mask_df.has_lesion]['volume_ml'].median()),
    'std_lesion_volume_ml': float(mask_df[mask_df.has_lesion]['volume_ml'].std()),
    'min_lesion_volume_ml': float(mask_df[mask_df.has_lesion]['volume_ml'].min()),
    'max_lesion_volume_ml': float(mask_df[mask_df.has_lesion]['volume_ml'].max()),
    'split': {k: {kk: int(vv) for kk, vv in v.items()} for k, v in split_summary.items()},
    'masks_analyzed': len(mask_df),
    'masks_with_lesion': int(mask_df.has_lesion.sum()),
}
with open(OUT_DIR / "dataset_stats.json", 'w') as f:
    json.dump(dataset_stats, f, indent=2)
print(f"\n    Stats saved to {OUT_DIR}/dataset_stats.json")

# ─── 6. Figure 1: Dataset Overview ───────────────────────────────────────────
print("\n[5] Generating Figure 1: Dataset Overview...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Figure 1 — Dataset Characteristics', fontsize=13, fontweight='bold', y=1.02)

# 1a: Class distribution
ax = axes[0]
labels = ['Negative\n(No Stroke)', 'Ischemic\nStroke', 'Hemorrhagic\nStroke']
counts = [(df.stroke==0).sum(), (df.stroke_type=='İSKEMİ').sum(), (df.stroke_type=='KANAMA').sum()]
colors = ['#2196F3', '#FF9800', '#F44336']
bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Number of CT Slices', fontsize=11)
ax.set_title('(a) Class Distribution', fontsize=11)
ax.set_ylim(0, max(counts) * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)

# 1b: Lesion volume distribution
ax = axes[1]
vol_data = mask_df[mask_df.has_lesion]['volume_ml']
ax.hist(vol_data, bins=30, color='#F44336', edgecolor='white', alpha=0.85)
ax.axvline(vol_data.mean(), color='#212121', linestyle='--', linewidth=1.5, label=f'Mean: {vol_data.mean():.1f} mL')
ax.axvline(vol_data.median(), color='#757575', linestyle=':', linewidth=1.5, label=f'Median: {vol_data.median():.1f} mL')
ax.set_xlabel('Estimated Lesion Volume (mL)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('(b) Lesion Volume Distribution\n(Hemorrhagic Cases)', fontsize=11)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 1c: Sample CT + Overlay
ax = axes[2]
# Pick a good example with lesion
sample_masks = mask_df[mask_df.has_lesion & (mask_df.lesion_pct > 2)].head(1)
if len(sample_masks) > 0:
    sample_id = sample_masks.iloc[0]['image_id']
    img_path = ISKEMI_PNG / f"{sample_id}.png"
    mask_path = ISKEMI_MASK / f"{sample_id}.png"
    img = np.array(Image.open(img_path).convert('L'))
    mask = np.array(Image.open(mask_path).convert('L'))
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    masked = np.ma.masked_where(mask < 20, mask)
    ax.imshow(masked, cmap='Reds', alpha=0.6, vmin=20, vmax=255)
    ax.set_title('(c) Sample CT with\nSegmentation Overlay', fontsize=11)
    ax.axis('off')
    ax.text(5, 505, f'ID: {sample_id}', color='white', fontsize=8)

plt.tight_layout()
fig.savefig(FIG_DIR / "figure1_dataset_overview.png", dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close()
print(f"    Saved: {FIG_DIR}/figure1_dataset_overview.png")

print("\n✓ Data pipeline complete.")
print(f"  Dataset: {dataset_stats['total_slices']:,} slices | "
      f"{dataset_stats['n_negative']:,} negative | "
      f"{dataset_stats['n_ischemic']:,} ischemic | "
      f"{dataset_stats['n_hemorrhagic']:,} hemorrhagic")
