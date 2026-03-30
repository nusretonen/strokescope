"""
StrokeScope - Model Training & Evaluation
Multi-task CNN: stroke classification + lesion segmentation + uncertainty
Uses real Teknofest 2021 dataset images.
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             classification_report)
from sklearn.calibration import calibration_curve
from scipy import stats

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE     = Path("/tmp/stroke_dataset")
OUT_DIR  = Path("/Users/nusretonen/stroke-paper/results")
FIG_DIR  = Path("/Users/nusretonen/stroke-paper/figures")
CODE_DIR = Path("/Users/nusretonen/stroke-paper/code")

KANAMA_PNG    = BASE / "kanama3" / "Kanama Veri Seti" / "PNG"
KANAMA_MASK   = BASE / "kanama3" / "Kanama Veri Seti" / "OVERLAY"
NEG_PNG       = BASE / "inmeyok3" / "İnme Yok Veri Set_PNG"
CHRONIC_PNG   = BASE / "iskemi3" / "İnme Yok Veri Set_PNG"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")
print("=" * 60)
print("StrokeScope - Model Training & Evaluation")
print("=" * 60)

# ─── Dataset ──────────────────────────────────────────────────────────────────
class StrokeDataset(Dataset):
    """
    Loads CT slices + optional segmentation mask.
    Labels: 0=negative, 1=ischemic, 2=hemorrhagic
    """
    IMG_SIZE = 224

    def __init__(self, samples, augment=False):
        self.samples = samples  # list of dict: {path, mask_path, label, has_mask}
        self.augment = augment
        self.tf_base = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
        ])
        self.tf_aug = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.3),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load image (convert RGBA->L->RGB for model)
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            img = Image.open(s['path']).convert('L').convert('RGB')
        except Exception:
            img = Image.new('RGB', (self.IMG_SIZE, self.IMG_SIZE), 0)
        tf = self.tf_aug if self.augment else self.tf_base
        img_t = tf(img)

        # Load mask if available
        if s['has_mask'] and s['mask_path'] is not None:
            try:
                mask = Image.open(s['mask_path']).convert('L')
                mask = mask.resize((self.IMG_SIZE, self.IMG_SIZE), Image.NEAREST)
                mask_t = torch.from_numpy(np.array(mask) > 20).float()
            except Exception:
                mask_t = torch.zeros(self.IMG_SIZE, self.IMG_SIZE)
        else:
            mask_t = torch.zeros(self.IMG_SIZE, self.IMG_SIZE)

        return img_t, s['label'], mask_t, int(s['has_mask'])


def build_samples():
    """Build sample list from available data."""
    samples = []

    # Negative samples (no stroke) - use all 4427 from inmeyok3
    neg_files = sorted(NEG_PNG.glob("*.png"))
    for p in neg_files:
        samples.append({'path': p, 'mask_path': None, 'label': 0, 'has_mask': False})

    # Hemorrhagic samples with masks
    hem_files = sorted(KANAMA_PNG.glob("*.png"))
    for p in hem_files:
        mask_p = KANAMA_MASK / p.name
        has_mask = mask_p.exists()
        samples.append({'path': p, 'mask_path': mask_p if has_mask else None,
                        'label': 2, 'has_mask': has_mask})

    # Chronic/other negatives from iskemi3 folder (labeled as negative in metadata)
    chr_files = sorted(CHRONIC_PNG.glob("*.png"))
    for p in chr_files:
        samples.append({'path': p, 'mask_path': None, 'label': 0, 'has_mask': False})

    np.random.seed(42)
    np.random.shuffle(samples)

    print(f"\n[1] Sample counts:")
    labels = [s['label'] for s in samples]
    for l, name in [(0, 'Negative'), (1, 'Ischemic'), (2, 'Hemorrhagic')]:
        print(f"    {name}: {labels.count(l):,}")
    print(f"    Total: {len(samples):,}")

    return samples


def split_samples(samples, train_frac=0.70, val_frac=0.10):
    """Stratified split."""
    from collections import defaultdict
    by_label = defaultdict(list)
    for s in samples:
        by_label[s['label']].append(s)

    train, val, test = [], [], []
    for label, items in by_label.items():
        n = len(items)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


# ─── Model ────────────────────────────────────────────────────────────────────
class StrokeScope(nn.Module):
    """
    Multi-task model:
    - Classification head: 3-class (neg / ischemic / hemorrhagic)
    - Segmentation head: binary lesion mask (224x224)
    - Uncertainty head: scalar uncertainty score per image
    """
    def __init__(self, n_classes=3):
        super().__init__()
        # Encoder: MobileNetV3 Small (fast, runs on CPU/MPS)
        backbone = models.mobilenet_v3_small(pretrained=True)
        self.encoder = backbone.features  # output: [B, 576, 7, 7]

        enc_ch = 576

        # Classification branch
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_ch, 128),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

        # Segmentation branch (lightweight decoder)
        self.seg_up1 = nn.Sequential(
            nn.ConvTranspose2d(enc_ch, 128, 2, stride=2),  # 14x14
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.seg_up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),       # 28x28
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.seg_up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),        # 56x56
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.seg_up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),        # 112x112
            nn.BatchNorm2d(16), nn.ReLU()
        )
        self.seg_up5 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 2, stride=2),         # 224x224
            nn.BatchNorm2d(8), nn.ReLU()
        )
        self.seg_out = nn.Conv2d(8, 1, 1)

        # Uncertainty head (per-image scalar)
        self.unc_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(enc_ch, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)                   # [B, 576, 7, 7]
        cls_logits = self.cls_head(self.cls_pool(feat))
        seg = self.seg_out(self.seg_up5(self.seg_up4(self.seg_up3(
              self.seg_up2(self.seg_up1(feat))))))  # [B, 1, 224, 224]
        unc = self.unc_head(feat)                # [B, 1]
        return cls_logits, seg.squeeze(1), unc.squeeze(1)


# ─── Loss ─────────────────────────────────────────────────────────────────────
def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(-1, -2))
    union = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
    return 1 - (2 * inter + eps) / (union + eps)


def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()


def combined_loss(cls_logits, seg_pred, unc_pred, labels, masks, has_mask,
                  lam1=1.0, lam2=0.5, lam3=0.3):
    # Classification loss
    loss_cls = focal_loss(cls_logits, labels)

    # Segmentation loss (only for samples with masks)
    mask_idx = has_mask.bool()
    if mask_idx.sum() > 0:
        seg_masked = seg_pred[mask_idx]
        gt_masked   = masks[mask_idx]
        bce = F.binary_cross_entropy_with_logits(seg_masked, gt_masked)
        dl  = dice_loss(seg_masked, gt_masked).mean()
        loss_seg = bce + dl
    else:
        loss_seg = torch.tensor(0.0, device=cls_logits.device)

    # Uncertainty: encourage higher uncertainty for harder samples
    # Proxy: uncertainty should correlate with classification confidence gap
    probs = F.softmax(cls_logits, dim=1)
    max_prob = probs.max(dim=1).values
    difficulty = 1 - max_prob  # harder = more uncertain
    loss_unc = F.mse_loss(unc_pred, difficulty.detach())

    return lam1 * loss_cls + lam2 * loss_seg + lam3 * loss_unc, \
           loss_cls.item(), loss_seg.item(), loss_unc.item()


# ─── Training ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler=None):
    model.train()
    total_loss = cls_l = seg_l = unc_l = 0
    correct = total = 0

    for imgs, labels, masks, has_mask in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        masks, has_mask = masks.to(DEVICE), has_mask.to(DEVICE)

        optimizer.zero_grad()
        cls_logits, seg_pred, unc_pred = model(imgs)
        loss, lc, ls, lu = combined_loss(cls_logits, seg_pred, unc_pred,
                                          labels, masks, has_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        cls_l += lc; seg_l += ls; unc_l += lu
        preds = cls_logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    if scheduler:
        scheduler.step()
    n = len(loader)
    return {'loss': total_loss/n, 'cls': cls_l/n, 'seg': seg_l/n,
            'unc': unc_l/n, 'acc': correct/total}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels, all_probs, all_unc = [], [], [], []
    all_seg_dice = []

    for imgs, labels, masks, has_mask in loader:
        imgs = imgs.to(DEVICE)
        cls_logits, seg_pred, unc_pred = model(imgs)
        probs = F.softmax(cls_logits, dim=1)

        all_logits.append(cls_logits.cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())
        all_unc.append(unc_pred.cpu())

        # Dice for masked samples
        mask_idx = has_mask.bool()
        if mask_idx.sum() > 0:
            seg_bin = (torch.sigmoid(seg_pred[mask_idx].cpu()) > 0.5).float()
            gt = masks[mask_idx]
            inter = (seg_bin * gt).sum(dim=(-1, -2))
            union = seg_bin.sum(dim=(-1, -2)) + gt.sum(dim=(-1, -2))
            dice = (2 * inter / (union + 1e-6)).numpy()
            all_seg_dice.extend(dice.tolist())

    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()
    all_unc    = torch.cat(all_unc).numpy()
    all_preds  = all_probs.argmax(1)

    # Binary stroke classification (1+2 vs 0)
    binary_labels = (all_labels > 0).astype(int)
    binary_probs  = all_probs[:, 1] + all_probs[:, 2]

    auroc = roc_auc_score(binary_labels, binary_probs)
    fpr, tpr, thresholds = roc_curve(binary_labels, binary_probs)

    # Youden's J threshold
    j_idx = np.argmax(tpr - fpr)
    opt_thresh = thresholds[j_idx]
    binary_preds = (binary_probs >= opt_thresh).astype(int)

    sens = recall_score(binary_labels, binary_preds)
    spec = precision_score(1 - binary_labels, 1 - binary_preds, zero_division=0)
    ppv  = precision_score(binary_labels, binary_preds, zero_division=0)
    npv  = precision_score(1 - binary_labels, 1 - binary_preds, zero_division=0)
    f1   = f1_score(binary_labels, binary_preds)

    # Hemorrhage-specific (label 2 vs rest)
    hem_labels = (all_labels == 2).astype(int)
    if hem_labels.sum() > 0:
        hem_probs = all_probs[:, 2]
        hem_auroc = roc_auc_score(hem_labels, hem_probs)
        hem_preds = (hem_probs >= 0.5).astype(int)
        hem_sens  = recall_score(hem_labels, hem_preds, zero_division=0)
    else:
        hem_auroc = hem_sens = 0.0

    metrics = {
        'auroc': float(auroc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'ppv': float(ppv),
        'npv': float(npv),
        'f1': float(f1),
        'opt_threshold': float(opt_thresh),
        'hem_auroc': float(hem_auroc),
        'hem_sensitivity': float(hem_sens),
        'mean_dice': float(np.mean(all_seg_dice)) if all_seg_dice else 0.0,
        'std_dice': float(np.std(all_seg_dice)) if all_seg_dice else 0.0,
        'n_dice_samples': len(all_seg_dice),
        'mean_uncertainty': float(all_unc.mean()),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'all_labels': all_labels.tolist(),
        'all_probs': all_probs.tolist(),
        'all_unc': all_unc.tolist(),
    }
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Build dataset
    samples = build_samples()
    train_s, val_s, test_s = split_samples(samples)
    print(f"\n[2] Splits: train={len(train_s):,} | val={len(val_s):,} | test={len(test_s):,}")

    # Dataloaders (num_workers=0 for macOS MPS stability)
    BATCH = 32
    train_loader = DataLoader(StrokeDataset(train_s, augment=True), batch_size=BATCH,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(StrokeDataset(val_s),   batch_size=BATCH,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(StrokeDataset(test_s),  batch_size=BATCH,
                              num_workers=0, pin_memory=False)

    # Model
    model = StrokeScope(n_classes=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[3] Model: StrokeScope | Params: {n_params/1e6:.2f}M | Device: {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

    # Training loop
    EPOCHS = 30
    print(f"\n[4] Training for {EPOCHS} epochs...")
    history = {'train_loss': [], 'val_auroc': [], 'val_dice': [], 'train_acc': []}
    best_auroc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, scheduler)
        vm = evaluate(model, val_loader)

        history['train_loss'].append(tr['loss'])
        history['val_auroc'].append(vm['auroc'])
        history['val_dice'].append(vm['mean_dice'])
        history['train_acc'].append(tr['acc'])

        if vm['auroc'] > best_auroc:
            best_auroc = vm['auroc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Ep {epoch:2d}/{EPOCHS} | loss={tr['loss']:.4f} acc={tr['acc']:.3f} | "
                  f"val AUROC={vm['auroc']:.4f} Dice={vm['mean_dice']:.3f} | {elapsed:.1f}s")

    # Load best model
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print(f"\n  Best val AUROC: {best_auroc:.4f}")

    # Final test evaluation
    print("\n[5] Test set evaluation...")
    test_metrics = evaluate(model, test_loader)

    print(f"\n  ── Test Results ──────────────────────────────────")
    print(f"  AUROC:       {test_metrics['auroc']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  PPV:         {test_metrics['ppv']:.4f}")
    print(f"  NPV:         {test_metrics['npv']:.4f}")
    print(f"  F1:          {test_metrics['f1']:.4f}")
    print(f"  Hemorrhage AUROC: {test_metrics['hem_auroc']:.4f}")
    print(f"  Hemorrhage Sens:  {test_metrics['hem_sensitivity']:.4f}")
    print(f"  Seg Dice:    {test_metrics['mean_dice']:.4f} ± {test_metrics['std_dice']:.4f}")
    print(f"  (n={test_metrics['n_dice_samples']} masked slices)")

    # ─── Volumetric Analysis ─────────────────────────────────────────────────
    print("\n[6] Volumetric analysis...")
    model.eval()
    vol_records = []

    hem_test = [s for s in test_s if s['label'] == 2 and s['has_mask']]

    with torch.no_grad():
        for s in hem_test[:100]:
            try:
                img = Image.open(s['path']).convert('L').convert('RGB')
                tf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
                img_t = tf(img).unsqueeze(0).to(DEVICE)
                _, seg_pred, _ = model(img_t)
                seg_bin = (torch.sigmoid(seg_pred[0]).cpu().numpy() > 0.5).astype(float)
                pred_px = seg_bin.sum()

                mask_arr = np.array(Image.open(s['mask_path']).convert('L').resize((224, 224), Image.NEAREST))
                gt_px = (mask_arr > 20).sum()

                # Volume in mL (voxel: 0.5x0.5mm in-plane, 5mm slice, downscaled 512->224)
                scale = (512/224)**2
                voxel_ml = 0.5 * 0.5 * 5.0 * scale / 1000.0
                pred_vol = float(pred_px * voxel_ml)
                gt_vol   = float(gt_px * voxel_ml)

                vol_records.append({'pred_vol': pred_vol, 'gt_vol': gt_vol,
                                    'diff': pred_vol - gt_vol,
                                    'abs_diff': abs(pred_vol - gt_vol)})
            except Exception:
                continue

    vol_df = pd.DataFrame(vol_records)
    if len(vol_df) > 0:
        mae_vol  = vol_df['abs_diff'].mean()
        iqr_vol  = vol_df['abs_diff'].quantile([0.25, 0.75]).values
        ba_mean  = vol_df['diff'].mean()
        ba_std   = vol_df['diff'].std()
        ba_loa_lo = ba_mean - 1.96 * ba_std
        ba_loa_hi = ba_mean + 1.96 * ba_std

        # Volume category accuracy (< 10, 10–70, >70 mL)
        def vol_cat(v):
            return 0 if v < 10 else (1 if v < 70 else 2)
        vol_df['pred_cat'] = vol_df['pred_vol'].map(vol_cat)
        vol_df['gt_cat']   = vol_df['gt_vol'].map(vol_cat)
        vol_cat_acc = (vol_df['pred_cat'] == vol_df['gt_cat']).mean()

        print(f"  MAE volume: {mae_vol:.1f} mL (IQR: {iqr_vol[0]:.1f}–{iqr_vol[1]:.1f})")
        print(f"  Bland-Altman: bias={ba_mean:.2f} mL, LoA=[{ba_loa_lo:.1f}, {ba_loa_hi:.1f}]")
        print(f"  Volume category accuracy: {vol_cat_acc:.3f}")
    else:
        mae_vol = iqr_vol = ba_mean = ba_std = ba_loa_lo = ba_loa_hi = vol_cat_acc = 0
        vol_df = pd.DataFrame({'pred_vol': [0], 'gt_vol': [0], 'diff': [0], 'abs_diff': [0]})

    # ─── Uncertainty Analysis ─────────────────────────────────────────────────
    print("\n[7] Uncertainty & human-in-the-loop simulation...")
    all_labels = np.array(test_metrics['all_labels'])
    all_probs  = np.array(test_metrics['all_probs'])
    all_unc    = np.array(test_metrics['all_unc'])

    binary_labels = (all_labels > 0).astype(int)
    binary_probs  = all_probs[:, 1] + all_probs[:, 2]
    binary_preds  = (binary_probs >= test_metrics['opt_threshold']).astype(int)

    # Tier assignment
    q33 = np.percentile(all_unc, 33)
    q67 = np.percentile(all_unc, 67)
    tiers = np.where(all_unc < q33, 'Low',
             np.where(all_unc < q67, 'Medium', 'High'))

    # Baseline FNR
    fn_base = ((binary_preds == 0) & (binary_labels == 1)).sum()
    fnr_base = fn_base / max(binary_labels.sum(), 1)

    # Human-in-loop: high-uncertainty positives corrected
    hitl_preds = binary_preds.copy()
    high_idx = tiers == 'High'
    hitl_preds[high_idx] = binary_labels[high_idx]  # simulate perfect radiologist
    fn_hitl = ((hitl_preds == 0) & (binary_labels == 1)).sum()
    fnr_hitl = fn_hitl / max(binary_labels.sum(), 1)
    pct_routed = high_idx.mean()

    print(f"  Baseline FNR: {fnr_base:.4f} ({fn_base} false negatives)")
    print(f"  After HITL routing ({pct_routed*100:.1f}% cases): FNR={fnr_hitl:.4f}")
    print(f"  FNR reduction: {(fnr_base - fnr_hitl)/fnr_base*100:.1f}%")

    # Save all results
    results = {
        'test_metrics': {k: v for k, v in test_metrics.items()
                         if k not in ('fpr', 'tpr', 'thresholds', 'all_labels', 'all_probs', 'all_unc')},
        'volumetry': {
            'mae_ml': float(mae_vol),
            'iqr_lo_ml': float(iqr_vol[0]) if hasattr(iqr_vol, '__len__') else 0,
            'iqr_hi_ml': float(iqr_vol[1]) if hasattr(iqr_vol, '__len__') else 0,
            'ba_bias_ml': float(ba_mean),
            'ba_loa_lo': float(ba_loa_lo),
            'ba_loa_hi': float(ba_loa_hi),
            'vol_cat_accuracy': float(vol_cat_acc),
            'n_volumetry_samples': len(vol_records),
        },
        'hitl': {
            'fnr_baseline': float(fnr_base),
            'fnr_hitl_high_only': float(fnr_hitl),
            'pct_routed_high': float(pct_routed),
            'fnr_reduction_pct': float((fnr_base - fnr_hitl) / max(fnr_base, 1e-9) * 100),
        },
        'history': history,
    }
    with open(OUT_DIR / 'model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    np.save(OUT_DIR / 'roc_data.npy', {
        'fpr': np.array(test_metrics['fpr']),
        'tpr': np.array(test_metrics['tpr']),
        'auroc': test_metrics['auroc']
    }, allow_pickle=True)
    vol_df.to_csv(OUT_DIR / 'volumetry_data.csv', index=False)
    np.save(OUT_DIR / 'uncertainty_data.npy', {
        'labels': all_labels, 'probs': all_probs,
        'unc': all_unc, 'tiers': tiers
    }, allow_pickle=True)

    print(f"\n✓ Results saved to {OUT_DIR}")
    return results, test_metrics, vol_df, all_labels, all_probs, all_unc, tiers, history


if __name__ == '__main__':
    results, test_metrics, vol_df, all_labels, all_probs, all_unc, tiers, history = main()
    print("\n✓ Training & evaluation complete.")
