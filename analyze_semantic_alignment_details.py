import os
import yaml
import torch
import numpy as np
import scipy.io as sio
import tifffile as tiff
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from load_model import load_encoder

os.makedirs("images/cross_dataset", exist_ok=True)
os.makedirs("results", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==========================================
# 1. Dataset Loaders & Helpers
# ==========================================

def load_indian_pines():
    data = sio.loadmat("data/indian_pines/Indian_pines.mat")
    gt = sio.loadmat("data/indian_pines/Indian_pines_gt.mat")
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_botswana():
    data = sio.loadmat("data/botswana/Botswana.mat")
    gt = sio.loadmat("data/botswana/Botswana_gt.mat")
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_pavia_university():
    data = sio.loadmat("data/pavia_university/Pavia.mat")
    gt = sio.loadmat("data/pavia_university/Pavia_gt.mat")
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_pavia_centre():
    data = sio.loadmat("data/pavia_centre/PaviaCentre.mat")
    gt = sio.loadmat("data/pavia_centre/PaviaCentre_gt.mat")
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_hyrank():
    img = tiff.imread("data/hyrank/TrainingSet/Dioni.tif")
    gt = tiff.imread("data/hyrank/TrainingSet/Dioni_GT.tif")
    if img.ndim == 3 and img.shape[0] < img.shape[2]:
        img = img.transpose(1, 2, 0)
    return img.astype(np.float32), gt.astype(np.int32)

def preprocess_hsi(X_pixels):
    p1 = np.percentile(X_pixels, 1, axis=0)
    p99 = np.percentile(X_pixels, 99, axis=0)
    X_clipped = np.clip(X_pixels, p1, p99)
    mean = np.mean(X_clipped, axis=0)
    std = np.std(X_clipped, axis=0)
    X_norm = (X_clipped - mean) / (std + 1e-8)
    return X_norm

def extract_embeddings(X_norm, band_dim, wavelengths=None, device="cpu"):
    wl_tensor = None
    if wavelengths is not None:
        wl_tensor = torch.tensor(wavelengths, dtype=torch.float32)
    encoder = load_encoder(band_dim=band_dim, wavelengths=wl_tensor, device=device)
    encoder.eval()
    
    batch_size = 2048
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):
            batch = torch.tensor(X_norm[i:i+batch_size], dtype=torch.float32, device=device)
            feats = encoder(batch)
            embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def pad_features(X, target_dim=220):
    if X.shape[1] == target_dim:
        return X
    padded = np.zeros((X.shape[0], target_dim))
    padded[:, :X.shape[1]] = X
    return padded

DATASET_COLORS = {
    0: "tab:blue",    # Indian Pines
    1: "tab:purple",  # Botswana
    2: "tab:orange",  # Pavia University
    3: "tab:red",     # Pavia Centre
    4: "tab:cyan"     # HyRank
}
DATASET_NAMES = {0: "Indian Pines", 1: "Botswana", 2: "Pavia University", 3: "Pavia Centre", 4: "HyRank"}

MAX_BANDS = 220

# ==========================================
# 2. Extract Data for Subgroups
# ==========================================

print("Extracting subgroup data...")

subgroup_raw = {0: [], 1: [], 2: [], 3: []} # Water, Trees, Soils, Urban
subgroup_emb = {0: [], 1: [], 2: [], 3: []}
subgroup_ds = {0: [], 1: [], 2: [], 3: []}

# Sample size configurations
limit_sem = {0: 500, 1: 500, 2: 500, 3: 500}

# 1. Indian Pines
img, gt = load_indian_pines()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
# Trees=6,14; Urban=16
mask_trees = (y_flat == 6) | (y_flat == 14)
mask_urban = (y_flat == 16)
for sem, mask in [(1, mask_trees), (3, mask_urban)]:
    idx = np.where(mask)[0]
    if len(idx) > 0:
        np.random.seed(42)
        s_idx = np.random.choice(idx, min(limit_sem[sem], len(idx)), replace=False)
        X_sub = X_flat[s_idx]
        X_sub_norm = preprocess_hsi(X_sub)
        X_sub_emb = StandardScaler().fit_transform(extract_embeddings(X_sub_norm, img.shape[2], device=DEVICE))
        subgroup_raw[sem].append(pad_features(X_sub_norm))
        subgroup_emb[sem].append(X_sub_emb)
        subgroup_ds[sem].append(np.full(len(s_idx), 0))

# 2. Botswana
img, gt = load_botswana()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
# Water=1; Trees=6,9; Soils=14
mask_water = (y_flat == 1)
mask_trees = (y_flat == 6) | (y_flat == 9)
mask_soils = (y_flat == 14)
for sem, mask in [(0, mask_water), (1, mask_trees), (2, mask_soils)]:
    idx = np.where(mask)[0]
    if len(idx) > 0:
        np.random.seed(42)
        s_idx = np.random.choice(idx, min(limit_sem[sem], len(idx)), replace=False)
        X_sub = X_flat[s_idx]
        X_sub_norm = preprocess_hsi(X_sub)
        X_sub_emb = StandardScaler().fit_transform(extract_embeddings(X_sub_norm, img.shape[2], device=DEVICE))
        subgroup_raw[sem].append(pad_features(X_sub_norm))
        subgroup_emb[sem].append(X_sub_emb)
        subgroup_ds[sem].append(np.full(len(s_idx), 1))

# 3. Pavia University
img, gt = load_pavia_university()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
# Trees=4; Soils=6; Urban=1,3,5,7,8
mask_trees = (y_flat == 4)
mask_soils = (y_flat == 6)
mask_urban = (y_flat == 1) | (y_flat == 3) | (y_flat == 5) | (y_flat == 7) | (y_flat == 8)
pu_waves = np.linspace(430.0, 860.0, img.shape[2])
for sem, mask in [(1, mask_trees), (2, mask_soils), (3, mask_urban)]:
    idx = np.where(mask)[0]
    if len(idx) > 0:
        np.random.seed(42)
        s_idx = np.random.choice(idx, min(limit_sem[sem], len(idx)), replace=False)
        X_sub = X_flat[s_idx]
        X_sub_norm = preprocess_hsi(X_sub)
        X_sub_emb = StandardScaler().fit_transform(extract_embeddings(X_sub_norm, img.shape[2], pu_waves, device=DEVICE))
        subgroup_raw[sem].append(pad_features(X_sub_norm))
        subgroup_emb[sem].append(X_sub_emb)
        subgroup_ds[sem].append(np.full(len(s_idx), 2))

# 4. Pavia Centre
img, gt = load_pavia_centre()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
# Water=1; Trees=2; Soils=9; Urban=3,4,5,6
mask_water = (y_flat == 1)
mask_trees = (y_flat == 2)
mask_soils = (y_flat == 9)
mask_urban = (y_flat == 3) | (y_flat == 4) | (y_flat == 5) | (y_flat == 6)
pc_waves = np.linspace(430.0, 860.0, img.shape[2])
for sem, mask in [(0, mask_water), (1, mask_trees), (2, mask_soils), (3, mask_urban)]:
    idx = np.where(mask)[0]
    if len(idx) > 0:
        np.random.seed(42)
        s_idx = np.random.choice(idx, min(limit_sem[sem], len(idx)), replace=False)
        X_sub = X_flat[s_idx]
        X_sub_norm = preprocess_hsi(X_sub)
        X_sub_emb = StandardScaler().fit_transform(extract_embeddings(X_sub_norm, img.shape[2], pc_waves, device=DEVICE))
        subgroup_raw[sem].append(pad_features(X_sub_norm))
        subgroup_emb[sem].append(X_sub_emb)
        subgroup_ds[sem].append(np.full(len(s_idx), 3))

# 5. HyRank
img, gt = load_hyrank()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
# Water=13,14; Trees=4,5,6,7,8; Soils=11,12; Urban=1,2
mask_water = (y_flat == 13) | (y_flat == 14)
mask_trees = (y_flat == 4) | (y_flat == 5) | (y_flat == 6) | (y_flat == 7) | (y_flat == 8)
mask_soils = (y_flat == 11) | (y_flat == 12)
mask_urban = (y_flat == 1) | (y_flat == 2)

with open("data/hyrank/hyrank_satellite.yaml") as f:
    hyrank_cfg = yaml.safe_load(f)
hr_waves = np.array(hyrank_cfg["info"]["wavelengths"]) * 1000.0

for sem, mask in [(0, mask_water), (1, mask_trees), (2, mask_soils), (3, mask_urban)]:
    idx = np.where(mask)[0]
    if len(idx) > 0:
        np.random.seed(42)
        s_idx = np.random.choice(idx, min(limit_sem[sem], len(idx)), replace=False)
        X_sub = X_flat[s_idx]
        X_sub_norm = preprocess_hsi(X_sub)
        X_sub_emb = StandardScaler().fit_transform(extract_embeddings(X_sub_norm, img.shape[2], hr_waves, device=DEVICE))
        subgroup_raw[sem].append(pad_features(X_sub_norm))
        subgroup_emb[sem].append(X_sub_emb)
        subgroup_ds[sem].append(np.full(len(s_idx), 4))

SEMANTIC_LABELS = {0: "Water", 1: "Trees", 2: "Soils", 3: "Urban"}

# ==========================================
# 3. Compute Projections & Save for Subgroups
# ==========================================

print("Computing projections for subgroups...")

subgroup_results = {}

for sem, name in SEMANTIC_LABELS.items():
    print(f"Processing subgroup: {name}")
    X_raw = np.concatenate(subgroup_raw[sem], axis=0)
    X_emb = np.concatenate(subgroup_emb[sem], axis=0)
    ds_labels = np.concatenate(subgroup_ds[sem], axis=0)
    
    # 2D Projections
    pca_raw = PCA(n_components=2, random_state=42).fit_transform(X_raw)
    pca_emb = PCA(n_components=2, random_state=42).fit_transform(X_emb)
    
    tsne_raw = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca').fit_transform(X_raw)
    tsne_emb = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca').fit_transform(X_emb)
    
    # Calculate Dataset Silhouette (how much points group by dataset - lower is better for cross-domain mixing)
    s_raw_pca = silhouette_score(pca_raw, ds_labels)
    s_emb_pca = silhouette_score(pca_emb, ds_labels)
    s_raw_tsne = silhouette_score(tsne_raw, ds_labels)
    s_emb_tsne = silhouette_score(tsne_emb, ds_labels)
    
    subgroup_results[name] = {
        "raw_pca": s_raw_pca, "emb_pca": s_emb_pca,
        "raw_tsne": s_raw_tsne, "emb_tsne": s_emb_tsne
    }
    
    # Plotting 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Cross-Dataset Subgroup Alignment: {name} (Colored by Dataset Origin)", fontsize=14, fontweight='bold')
    
    def plot_sub(ax, coords, title):
        for ds_val, ds_name in DATASET_NAMES.items():
            mask = (ds_labels == ds_val)
            if np.sum(mask) > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1], color=DATASET_COLORS[ds_val], s=12, alpha=0.6, label=ds_name)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
    plot_sub(axes[0, 0], pca_raw, f"Raw Spectrum PCA\n(Dataset Silhouette: {s_raw_pca:.4f})")
    plot_sub(axes[0, 1], pca_emb, f"Hyperfocus Embedding PCA\n(Dataset Silhouette: {s_emb_pca:.4f})")
    plot_sub(axes[1, 0], tsne_raw, f"Raw Spectrum t-SNE\n(Dataset Silhouette: {s_raw_tsne:.4f})")
    plot_sub(axes[1, 1], tsne_emb, f"Hyperfocus Embedding t-SNE\n(Dataset Silhouette: {s_emb_tsne:.4f})")
    
    # Consolidated legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    plt.savefig(f"images/cross_dataset/semantic_alignment_{name.lower()}.png", dpi=140)
    plt.close()

# ==========================================
# 4. Case Study A: Clear Semantic Boundaries
# ==========================================

print("Executing Case Study A: Clear Semantic Boundaries...")
# Dataset: Pavia University
# Classes: Meadows (gt=2), Painted metal sheets (gt=5), Bare Soil (gt=6)
img, gt = load_pavia_university()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)

c1_mask = (y_flat == 2) # Meadows
c2_mask = (y_flat == 5) # Painted metal sheets
c3_mask = (y_flat == 6) # Bare Soil

pu_waves = np.linspace(430.0, 860.0, img.shape[2])
case_a_raw = []
case_a_emb = []
case_a_labels = []

# Colors: Meadows = green, Painted metal = cyan, Bare soil = brown
case_a_colors = {0: "forestgreen", 1: "deepskyblue", 2: "saddlebrown"}
case_a_names = {0: "Meadows (Vegetation)", 1: "Painted Metal Sheets", 2: "Bare Soil"}

for i, mask in enumerate([c1_mask, c2_mask, c3_mask]):
    idx = np.where(mask)[0]
    np.random.seed(42)
    s_idx = np.random.choice(idx, 500, replace=False)
    X_sub = X_flat[s_idx]
    X_sub_norm = preprocess_hsi(X_sub)
    X_sub_emb = StandardScaler().fit_transform(extract_embeddings(X_sub_norm, img.shape[2], pu_waves, device=DEVICE))
    case_a_raw.append(X_sub_norm)
    case_a_emb.append(X_sub_emb)
    case_a_labels.append(np.full(500, i))

X_raw_a = np.concatenate(case_a_raw, axis=0)
X_emb_a = np.concatenate(case_a_emb, axis=0)
y_a = np.concatenate(case_a_labels, axis=0)

pca_raw_a = PCA(n_components=2, random_state=42).fit_transform(X_raw_a)
pca_emb_a = PCA(n_components=2, random_state=42).fit_transform(X_emb_a)
tsne_raw_a = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca').fit_transform(X_raw_a)
tsne_emb_a = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca').fit_transform(X_emb_a)

s_a_raw_pca = silhouette_score(pca_raw_a, y_a)
s_a_emb_pca = silhouette_score(pca_emb_a, y_a)
s_a_raw_tsne = silhouette_score(tsne_raw_a, y_a)
s_a_emb_tsne = silhouette_score(tsne_emb_a, y_a)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Case Study A: Clear Semantic Boundaries (Pavia University)", fontsize=14, fontweight='bold')

def plot_case_a(ax, coords, title):
    for val, name in case_a_names.items():
        mask = (y_a == val)
        ax.scatter(coords[mask, 0], coords[mask, 1], color=case_a_colors[val], s=12, alpha=0.6, label=name)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

plot_case_a(axes[0, 0], pca_raw_a, f"Raw Spectrum PCA\n(Semantic Silhouette: {s_a_raw_pca:.4f})")
plot_case_a(axes[0, 1], pca_emb_a, f"Hyperfocus Embedding PCA\n(Semantic Silhouette: {s_a_emb_pca:.4f})")
plot_case_a(axes[1, 0], tsne_raw_a, f"Raw Spectrum t-SNE\n(Semantic Silhouette: {s_a_raw_tsne:.4f})")
plot_case_a(axes[1, 1], tsne_emb_a, f"Hyperfocus Embedding t-SNE\n(Semantic Silhouette: {s_a_emb_tsne:.4f})")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig("images/cross_dataset/case_study_clear_semantic.png", dpi=140)
plt.close()

# ==========================================
# 5. Case Study B: Sensor-Invariant Mixing
# ==========================================

print("Executing Case Study B: Sensor-Invariant Mixing...")
# Target: Pavia University & Pavia Centre
# Classes: Asphalt (PU Class 1 & PC Class 3), Trees (PU Class 4 & PC Class 2)
# Here we want to observe:
# 1. Semantic Separation (Asphalt vs Trees) -> Higher is better
# 2. Dataset Separation (Pavia U vs Pavia Centre) -> Lower is better (representing perfect alignment)

# Load Pavia University
img_pu, gt_pu = load_pavia_university()
X_flat_pu = img_pu.reshape(-1, img_pu.shape[2])
y_flat_pu = gt_pu.reshape(-1)

# Load Pavia Centre
img_pc, gt_pc = load_pavia_centre()
X_flat_pc = img_pc.reshape(-1, img_pc.shape[2])
y_flat_pc = gt_pc.reshape(-1)

pu_asphalt_idx = np.where(y_flat_pu == 1)[0]
pu_trees_idx = np.where(y_flat_pu == 4)[0]

pc_asphalt_idx = np.where(y_flat_pc == 3)[0]
pc_trees_idx = np.where(y_flat_pc == 2)[0]

case_b_raw = []
case_b_emb = []
case_b_sem_labels = [] # 0: Asphalt, 1: Trees
case_b_ds_labels = []  # 0: Pavia Univ, 1: Pavia Centre

# Sample pixels
np.random.seed(42)
s_pu_asphalt = np.random.choice(pu_asphalt_idx, 300, replace=False)
s_pu_trees = np.random.choice(pu_trees_idx, 300, replace=False)
s_pc_asphalt = np.random.choice(pc_asphalt_idx, 300, replace=False)
s_pc_trees = np.random.choice(pc_trees_idx, 300, replace=False)

# Extract Pavia U
X_pu_asphalt = X_flat_pu[s_pu_asphalt]
X_pu_asphalt_norm = preprocess_hsi(X_pu_asphalt)
X_pu_asphalt_emb = StandardScaler().fit_transform(extract_embeddings(X_pu_asphalt_norm, img_pu.shape[2], pu_waves, device=DEVICE))

X_pu_trees = X_flat_pu[s_pu_trees]
X_pu_trees_norm = preprocess_hsi(X_pu_trees)
X_pu_trees_emb = StandardScaler().fit_transform(extract_embeddings(X_pu_trees_norm, img_pu.shape[2], pu_waves, device=DEVICE))

# Extract Pavia C
X_pc_asphalt = X_flat_pc[s_pc_asphalt]
X_pc_asphalt_norm = preprocess_hsi(X_pc_asphalt)
X_pc_asphalt_emb = StandardScaler().fit_transform(extract_embeddings(X_pc_asphalt_norm, img_pc.shape[2], pc_waves, device=DEVICE))

X_pc_trees = X_flat_pc[s_pc_trees]
X_pc_trees_norm = preprocess_hsi(X_pc_trees)
X_pc_trees_emb = StandardScaler().fit_transform(extract_embeddings(X_pc_trees_norm, img_pc.shape[2], pc_waves, device=DEVICE))

# Add PU Asphalt
case_b_raw.append(pad_features(X_pu_asphalt_norm))
case_b_emb.append(X_pu_asphalt_emb)
case_b_sem_labels.append(np.full(300, 0))
case_b_ds_labels.append(np.full(300, 0))

# Add PU Trees
case_b_raw.append(pad_features(X_pu_trees_norm))
case_b_emb.append(X_pu_trees_emb)
case_b_sem_labels.append(np.full(300, 1))
case_b_ds_labels.append(np.full(300, 0))

# Add PC Asphalt
case_b_raw.append(pad_features(X_pc_asphalt_norm))
case_b_emb.append(X_pc_asphalt_emb)
case_b_sem_labels.append(np.full(300, 0))
case_b_ds_labels.append(np.full(300, 1))

# Add PC Trees
case_b_raw.append(pad_features(X_pc_trees_norm))
case_b_emb.append(X_pc_trees_emb)
case_b_sem_labels.append(np.full(300, 1))
case_b_ds_labels.append(np.full(300, 1))

X_raw_b = np.concatenate(case_b_raw, axis=0)
X_emb_b = np.concatenate(case_b_emb, axis=0)
y_sem_b = np.concatenate(case_b_sem_labels, axis=0)
y_ds_b = np.concatenate(case_b_ds_labels, axis=0)

# 2D t-SNE Projections
tsne_raw_b = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca').fit_transform(X_raw_b)
tsne_emb_b = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca').fit_transform(X_emb_b)

# PCA Projections
pca_raw_b = PCA(n_components=2, random_state=42).fit_transform(X_raw_b)
pca_emb_b = PCA(n_components=2, random_state=42).fit_transform(X_emb_b)

# Metrics
s_sem_raw = silhouette_score(tsne_raw_b, y_sem_b)
s_ds_raw = silhouette_score(tsne_raw_b, y_ds_b)
dai_raw = s_sem_raw - s_ds_raw

s_sem_emb = silhouette_score(tsne_emb_b, y_sem_b)
s_ds_emb = silhouette_score(tsne_emb_b, y_ds_b)
dai_emb = s_sem_emb - s_ds_emb

# 4-panel plot showing t-SNE comparison
# Top Row: Colored by Semantic (Asphalt vs Trees)
# Bottom Row: Colored by Dataset Origin (Pavia Univ vs Pavia Centre)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Case Study B: Cross-Dataset Alignment (Pavia University + Pavia Centre)", fontsize=14, fontweight='bold')

# Top Left: Raw t-SNE Semantic Coloring
axes[0, 0].scatter(tsne_raw_b[y_sem_b == 0, 0], tsne_raw_b[y_sem_b == 0, 1], color="darkgrey", s=12, alpha=0.6, label="Asphalt")
axes[0, 0].scatter(tsne_raw_b[y_sem_b == 1, 0], tsne_raw_b[y_sem_b == 1, 1], color="forestgreen", s=12, alpha=0.6, label="Trees")
axes[0, 0].set_title(f"Raw t-SNE (Semantic Coloring)\n(S_sem: {s_sem_raw:.4f})", fontsize=11, fontweight='bold')
axes[0, 0].grid(True, linestyle='--', alpha=0.5)
axes[0, 0].legend()

# Top Right: Emb t-SNE Semantic Coloring
axes[0, 1].scatter(tsne_emb_b[y_sem_b == 0, 0], tsne_emb_b[y_sem_b == 0, 1], color="darkgrey", s=12, alpha=0.6, label="Asphalt")
axes[0, 1].scatter(tsne_emb_b[y_sem_b == 1, 0], tsne_emb_b[y_sem_b == 1, 1], color="forestgreen", s=12, alpha=0.6, label="Trees")
axes[0, 1].set_title(f"Embedding t-SNE (Semantic Coloring)\n(S_sem: {s_sem_emb:.4f})", fontsize=11, fontweight='bold')
axes[0, 1].grid(True, linestyle='--', alpha=0.5)
axes[0, 1].legend()

# Bottom Left: Raw t-SNE Dataset Coloring
axes[1, 0].scatter(tsne_raw_b[y_ds_b == 0, 0], tsne_raw_b[y_ds_b == 0, 1], color="tab:orange", s=12, alpha=0.6, label="Pavia University")
axes[1, 0].scatter(tsne_raw_b[y_ds_b == 1, 0], tsne_raw_b[y_ds_b == 1, 1], color="tab:red", s=12, alpha=0.6, label="Pavia Centre")
axes[1, 0].set_title(f"Raw t-SNE (Dataset Coloring)\n(S_ds: {s_ds_raw:.4f})", fontsize=11, fontweight='bold')
axes[1, 0].grid(True, linestyle='--', alpha=0.5)
axes[1, 0].legend()

# Bottom Right: Emb t-SNE Dataset Coloring
axes[1, 1].scatter(tsne_emb_b[y_ds_b == 0, 0], tsne_emb_b[y_ds_b == 0, 1], color="tab:orange", s=12, alpha=0.6, label="Pavia University")
axes[1, 1].scatter(tsne_emb_b[y_ds_b == 1, 0], tsne_emb_b[y_ds_b == 1, 1], color="tab:red", s=12, alpha=0.6, label="Pavia Centre")
axes[1, 1].set_title(f"Embedding t-SNE (Dataset Coloring)\n(S_ds: {s_ds_emb:.4f})", fontsize=11, fontweight='bold')
axes[1, 1].grid(True, linestyle='--', alpha=0.5)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("images/cross_dataset/case_study_sensor_mix.png", dpi=140)
plt.close()

# ==========================================
# 6. Save All Detailed Statistics
# ==========================================

stats_path = "results/semantic_details_summary.txt"
with open(stats_path, "w", encoding="utf-8") as f:
    f.write("=== Detailed Semantic subgroup alignment metrics (Dataset Silhouette, S_ds) ===\n\n")
    f.write("| Subgroup | Raw PCA | Embedding PCA | Raw t-SNE | Embedding t-SNE |\n")
    f.write("|----------|---------|---------------|-----------|-----------------|\n")
    for name, res in subgroup_results.items():
        f.write(f"| {name:<8} | {res['raw_pca']:.4f}  | {res['emb_pca']:.4f}        | {res['raw_tsne']:.4f}    | {res['emb_tsne']:.4f}          |\n")
    
    f.write("\n\n=== Case Study A (Clear Semantic Boundaries - Pavia University) ===\n")
    f.write(f"Raw PCA S_sem: {s_a_raw_pca:.4f}\n")
    f.write(f"Embedding PCA S_sem: {s_a_emb_pca:.4f}\n")
    f.write(f"Raw t-SNE S_sem: {s_a_raw_tsne:.4f}\n")
    f.write(f"Embedding t-SNE S_sem: {s_a_emb_tsne:.4f}\n")
    
    f.write("\n\n=== Case Study B (Cross-Dataset Alignment - Pavia Univ + Centre t-SNE) ===\n")
    f.write(f"Raw t-SNE: S_sem = {s_sem_raw:.4f}, S_ds = {s_ds_raw:.4f}, DAI = {dai_raw:.4f}\n")
    f.write(f"Embedding t-SNE: S_sem = {s_sem_emb:.4f}, S_ds = {s_ds_emb:.4f}, DAI = {dai_emb:.4f}\n")

print(f"\nDetailed semantic analysis completed successfully! Metrics saved to {stats_path}")
