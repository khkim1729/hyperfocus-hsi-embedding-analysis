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
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from load_model import load_encoder

# Create directories
os.makedirs("images/cross_dataset", exist_ok=True)
os.makedirs("results", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==========================================
# 1. Dataset Loader Functions (Standardized)
# ==========================================

def load_indian_pines():
    data_path = "data/indian_pines/Indian_pines.mat"
    gt_path = "data/indian_pines/Indian_pines_gt.mat"
    data = sio.loadmat(data_path)
    gt = sio.loadmat(gt_path)
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_botswana():
    data_path = "data/botswana/Botswana.mat"
    gt_path = "data/botswana/Botswana_gt.mat"
    data = sio.loadmat(data_path)
    gt = sio.loadmat(gt_path)
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_pavia_university():
    data_path = "data/pavia_university/Pavia.mat"
    gt_path = "data/pavia_university/Pavia_gt.mat"
    data = sio.loadmat(data_path)
    gt = sio.loadmat(gt_path)
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_pavia_centre():
    data_path = "data/pavia_centre/PaviaCentre.mat"
    gt_path = "data/pavia_centre/PaviaCentre_gt.mat"
    data = sio.loadmat(data_path)
    gt = sio.loadmat(gt_path)
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_hyrank():
    data_path = "data/hyrank/TrainingSet/Dioni.tif"
    gt_path = "data/hyrank/TrainingSet/Dioni_GT.tif"
    img = tiff.imread(data_path)
    gt = tiff.imread(gt_path)
    if img.ndim == 3 and img.shape[0] < img.shape[2]:
        img = img.transpose(1, 2, 0)
    return img.astype(np.float32), gt.astype(np.int32)

# ==========================================
# 2. Preprocessing & Embedding Extraction
# ==========================================

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

def draw_confidence_ellipse(x, y, ax, color, n_std=1.2, **kwargs):
    if len(x) < 5:
        return
    cov = np.cov(x, y)
    std_x = np.sqrt(cov[0, 0])
    std_y = np.sqrt(cov[1, 1])
    if std_x < 1e-6 or std_y < 1e-6:
        return
        
    pearson = cov[0, 1] / (std_x * std_y)
    pearson = np.clip(pearson, -0.999, 0.999)
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor='none', edgecolor=color, linewidth=1.3, alpha=0.6, linestyle='--', **kwargs)
    
    scale_x = std_x * n_std
    mean_x = np.mean(x)
    
    scale_y = std_y * n_std
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
        
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

# ==========================================
# 3. Extraction and Grouping Logic
# ==========================================

print("Loading all datasets and grouping similar classes...")

# Target Semantic Mapping:
# 0: Water, 1: Trees, 2: Soils, 3: Urban
SEMANTIC_NAMES = {0: "Water", 1: "Trees", 2: "Soils", 3: "Urban"}
DATASET_NAMES = {0: "Indian Pines", 1: "Botswana", 2: "Pavia University", 3: "Pavia Centre", 4: "HyRank"}

# Storage for collected pixels
raw_data_list = []
emb_data_list = []
semantic_labels_list = []
dataset_labels_list = []

# Pad features helper to handle different band counts for raw spectrums
def pad_features(X, target_dim):
    if X.shape[1] == target_dim:
        return X
    padded = np.zeros((X.shape[0], target_dim))
    padded[:, :X.shape[1]] = X
    return padded

# Maximum bands is 220 (Indian Pines)
MAX_BANDS = 220

# 1. Indian Pines
img, gt = load_indian_pines()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
# Mapping: Trees=6,14; Urban=16
mask_trees = (y_flat == 6) | (y_flat == 14)
mask_urban = (y_flat == 16)
for sem_val, mask in [(1, mask_trees), (3, mask_urban)]:
    indices = np.where(mask)[0]
    if len(indices) > 0:
        # Sample max 500 pixels per semantic group to balance
        np.random.seed(42)
        sampled_idx = np.random.choice(indices, min(500, len(indices)), replace=False)
        X_sub = X_flat[sampled_idx]
        X_sub_norm = preprocess_hsi(X_sub)
        X_sub_emb = extract_embeddings(X_sub_norm, band_dim=img.shape[2], device=DEVICE)
        X_sub_emb = StandardScaler().fit_transform(X_sub_emb)
        
        raw_data_list.append(pad_features(X_sub_norm, MAX_BANDS))
        emb_data_list.append(X_sub_emb)
        semantic_labels_list.append(np.full(len(sampled_idx), sem_val))
        dataset_labels_list.append(np.full(len(sampled_idx), 0))

# 2. Botswana
img, gt = load_botswana()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
# Mapping: Water=1; Trees=6,9; Soils=14
mask_water = (y_flat == 1)
mask_trees = (y_flat == 6) | (y_flat == 9)
mask_soils = (y_flat == 14)
for sem_val, mask in [(0, mask_water), (1, mask_trees), (2, mask_soils)]:
    indices = np.where(mask)[0]
    if len(indices) > 0:
        np.random.seed(42)
        sampled_idx = np.random.choice(indices, min(300, len(indices)), replace=False)
        X_sub = X_flat[sampled_idx]
        X_sub_norm = preprocess_hsi(X_sub)
        X_sub_emb = extract_embeddings(X_sub_norm, band_dim=img.shape[2], device=DEVICE)
        X_sub_emb = StandardScaler().fit_transform(X_sub_emb)
        
        raw_data_list.append(pad_features(X_sub_norm, MAX_BANDS))
        emb_data_list.append(X_sub_emb)
        semantic_labels_list.append(np.full(len(sampled_idx), sem_val))
        dataset_labels_list.append(np.full(len(sampled_idx), 1))

# 3. Pavia University
img, gt = load_pavia_university()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
# Mapping: Trees=4; Soils=6; Urban=1,3,5,7,8
mask_trees = (y_flat == 4)
mask_soils = (y_flat == 6)
mask_urban = (y_flat == 1) | (y_flat == 3) | (y_flat == 5) | (y_flat == 7) | (y_flat == 8)
pu_waves = np.linspace(430.0, 860.0, img.shape[2])
for sem_val, mask in [(1, mask_trees), (2, mask_soils), (3, mask_urban)]:
    indices = np.where(mask)[0]
    if len(indices) > 0:
        np.random.seed(42)
        sampled_idx = np.random.choice(indices, min(400, len(indices)), replace=False)
        X_sub = X_flat[sampled_idx]
        X_sub_norm = preprocess_hsi(X_sub)
        X_sub_emb = extract_embeddings(X_sub_norm, band_dim=img.shape[2], wavelengths=pu_waves, device=DEVICE)
        X_sub_emb = StandardScaler().fit_transform(X_sub_emb)
        
        raw_data_list.append(pad_features(X_sub_norm, MAX_BANDS))
        emb_data_list.append(X_sub_emb)
        semantic_labels_list.append(np.full(len(sampled_idx), sem_val))
        dataset_labels_list.append(np.full(len(sampled_idx), 2))

# 4. Pavia Centre
img, gt = load_pavia_centre()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
# Mapping: Water=1; Trees=2; Soils=9; Urban=3,4,5,6
mask_water = (y_flat == 1)
mask_trees = (y_flat == 2)
mask_soils = (y_flat == 9)
mask_urban = (y_flat == 3) | (y_flat == 4) | (y_flat == 5) | (y_flat == 6)
pc_waves = np.linspace(430.0, 860.0, img.shape[2])
for sem_val, mask in [(0, mask_water), (1, mask_trees), (2, mask_soils), (3, mask_urban)]:
    indices = np.where(mask)[0]
    if len(indices) > 0:
        np.random.seed(42)
        sampled_idx = np.random.choice(indices, min(300, len(indices)), replace=False)
        X_sub = X_flat[sampled_idx]
        X_sub_norm = preprocess_hsi(X_sub)
        X_sub_emb = extract_embeddings(X_sub_norm, band_dim=img.shape[2], wavelengths=pc_waves, device=DEVICE)
        X_sub_emb = StandardScaler().fit_transform(X_sub_emb)
        
        raw_data_list.append(pad_features(X_sub_norm, MAX_BANDS))
        emb_data_list.append(X_sub_emb)
        semantic_labels_list.append(np.full(len(sampled_idx), sem_val))
        dataset_labels_list.append(np.full(len(sampled_idx), 3))

# 5. HyRank
img, gt = load_hyrank()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
# Mapping: Water=13,14; Trees=4,5,6,7,8; Soils=11,12; Urban=1,2
mask_water = (y_flat == 13) | (y_flat == 14)
mask_trees = (y_flat == 4) | (y_flat == 5) | (y_flat == 6) | (y_flat == 7) | (y_flat == 8)
mask_soils = (y_flat == 11) | (y_flat == 12)
mask_urban = (y_flat == 1) | (y_flat == 2)

with open("data/hyrank/hyrank_satellite.yaml") as f:
    hyrank_cfg = yaml.safe_load(f)
hr_waves = np.array(hyrank_cfg["info"]["wavelengths"]) * 1000.0

for sem_val, mask in [(0, mask_water), (1, mask_trees), (2, mask_soils), (3, mask_urban)]:
    indices = np.where(mask)[0]
    if len(indices) > 0:
        np.random.seed(42)
        sampled_idx = np.random.choice(indices, min(300, len(indices)), replace=False)
        X_sub = X_flat[sampled_idx]
        X_sub_norm = preprocess_hsi(X_sub)
        X_sub_emb = extract_embeddings(X_sub_norm, band_dim=img.shape[2], wavelengths=hr_waves, device=DEVICE)
        X_sub_emb = StandardScaler().fit_transform(X_sub_emb)
        
        raw_data_list.append(pad_features(X_sub_norm, MAX_BANDS))
        emb_data_list.append(X_sub_emb)
        semantic_labels_list.append(np.full(len(sampled_idx), sem_val))
        dataset_labels_list.append(np.full(len(sampled_idx), 4))

# Concatenate all collected samples
X_raw_all = np.concatenate(raw_data_list, axis=0)
X_emb_all = np.concatenate(emb_data_list, axis=0)
semantic_labels = np.concatenate(semantic_labels_list, axis=0)
dataset_labels = np.concatenate(dataset_labels_list, axis=0)

print(f"Total samples collected: {len(X_raw_all)}")
for sem_val, name in SEMANTIC_NAMES.items():
    print(f" - {name}: {np.sum(semantic_labels == sem_val)} pixels")
for ds_val, name in DATASET_NAMES.items():
    print(f" - {name}: {np.sum(dataset_labels == ds_val)} pixels")

# ==========================================
# 4. Dimensionality Reduction & Projection
# ==========================================

print("Computing PCA and t-SNE projections...")

# PCA Raw
pca_raw = PCA(n_components=2, random_state=42)
X_raw_pca = pca_raw.fit_transform(X_raw_all)

# PCA Embedding
pca_emb = PCA(n_components=2, random_state=42)
X_emb_pca = pca_emb.fit_transform(X_emb_all)

# t-SNE Raw (Unsupervised)
tsne_raw = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000, init='pca')
X_raw_tsne = tsne_raw.fit_transform(X_raw_all)

# t-SNE Embedding (Unsupervised)
tsne_emb = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000, init='pca')
X_emb_tsne = tsne_emb.fit_transform(X_emb_all)

# ==========================================
# 5. Metric Computation (Silhouette & DAI)
# ==========================================

print("Calculating clustering metrics...")

def compute_metrics(X_2d):
    s_sem = silhouette_score(X_2d, semantic_labels)
    s_ds = silhouette_score(X_2d, dataset_labels)
    dai = s_sem - s_ds
    return s_sem, s_ds, dai

metrics = {
    "raw_pca": compute_metrics(X_raw_pca),
    "emb_pca": compute_metrics(X_emb_pca),
    "raw_tsne": compute_metrics(X_raw_tsne),
    "emb_tsne": compute_metrics(X_emb_tsne)
}

# ==========================================
# 6. Visualization & Plotting
# ==========================================

print("Generating plots...")

# Styling
SEMANTIC_COLORS = {
    0: "royalblue",   # Water
    1: "forestgreen", # Trees
    2: "saddlebrown", # Soils
    3: "darkgrey"     # Urban
}

DATASET_COLORS = {
    0: "tab:blue",    # Indian Pines
    1: "tab:purple",  # Botswana
    2: "tab:orange",  # Pavia Univ
    3: "tab:red",     # Pavia Centre
    4: "tab:cyan"     # HyRank
}

# Plot 1: Colored by Semantic Class
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

def plot_semantic(ax, X_2d, title):
    for sem_val, name in SEMANTIC_NAMES.items():
        mask = (semantic_labels == sem_val)
        color = SEMANTIC_COLORS[sem_val]
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], color=color, s=12, alpha=0.55, label=name, zorder=2)
        # Draw confidence ellipse
        draw_confidence_ellipse(X_2d[mask, 0], X_2d[mask, 1], ax, color=color, n_std=1.2, zorder=3)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

plot_semantic(axes[0, 0], X_raw_pca, f"Raw Spectrum PCA\n(Sem Silhouette: {metrics['raw_pca'][0]:.4f})")
plot_semantic(axes[0, 1], X_emb_pca, f"Hyperfocus Embedding PCA\n(Sem Silhouette: {metrics['emb_pca'][0]:.4f})")
plot_semantic(axes[1, 0], X_raw_tsne, f"Raw Spectrum t-SNE\n(Sem Silhouette: {metrics['raw_tsne'][0]:.4f})")
plot_semantic(axes[1, 1], X_emb_tsne, f"Hyperfocus Embedding t-SNE\n(Sem Silhouette: {metrics['emb_tsne'][0]:.4f})")

# Consolidated Legend
handles, labels = [], []
for ax in axes.flat:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5), title="Semantic Classes", fontsize=10)

plt.tight_layout()
fig.subplots_adjust(right=0.88)
plt.savefig("images/cross_dataset/semantic_alignment_by_class.png", dpi=140, bbox_inches='tight')
plt.close()

# Plot 2: Colored by Dataset Origin
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

def plot_dataset(ax, X_2d, title):
    for ds_val, name in DATASET_NAMES.items():
        mask = (dataset_labels == ds_val)
        color = DATASET_COLORS[ds_val]
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], color=color, s=10, alpha=0.55, label=name)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

plot_dataset(axes[0, 0], X_raw_pca, f"Raw Spectrum PCA\n(Dataset Silhouette: {metrics['raw_pca'][1]:.4f})")
plot_dataset(axes[0, 1], X_emb_pca, f"Hyperfocus Embedding PCA\n(Dataset Silhouette: {metrics['emb_pca'][1]:.4f})")
plot_dataset(axes[1, 0], X_raw_tsne, f"Raw Spectrum t-SNE\n(Dataset Silhouette: {metrics['raw_tsne'][1]:.4f})")
plot_dataset(axes[1, 1], X_emb_tsne, f"Hyperfocus Embedding t-SNE\n(Dataset Silhouette: {metrics['emb_tsne'][1]:.4f})")

# Consolidated Legend
handles, labels = [], []
for ax in axes.flat:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5), title="Source Datasets", fontsize=10)

plt.tight_layout()
fig.subplots_adjust(right=0.88)
plt.savefig("images/cross_dataset/semantic_alignment_by_dataset.png", dpi=140, bbox_inches='tight')
plt.close()

# ==========================================
# 7. Write Summary Text Report
# ==========================================

summary_path = "results/semantic_alignment_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== Cross-Dataset Semantic Alignment Metrics ===\n\n")
    f.write(f"| Model & Space | Semantic Silhouette (S_sem) | Dataset Silhouette (S_ds) | Domain-Agnostic Index (DAI) |\n")
    f.write(f"|---------------|-----------------------------|---------------------------|----------------------------|\n")
    f.write(f"| Raw PCA       | {metrics['raw_pca'][0]:.4f}                       | {metrics['raw_pca'][1]:.4f}                     | {metrics['raw_pca'][2]:.4f}                     |\n")
    f.write(f"| Emb PCA       | {metrics['emb_pca'][0]:.4f}                       | {metrics['emb_pca'][1]:.4f}                     | {metrics['emb_pca'][2]:.4f}                     |\n")
    f.write(f"| Raw t-SNE     | {metrics['raw_tsne'][0]:.4f}                       | {metrics['raw_tsne'][1]:.4f}                     | {metrics['raw_tsne'][2]:.4f}                     |\n")
    f.write(f"| Emb t-SNE     | {metrics['emb_tsne'][0]:.4f}                       | {metrics['emb_tsne'][1]:.4f}                     | {metrics['emb_tsne'][2]:.4f}                     |\n")

print(f"\nSemantic alignment completed successfully! Metrics saved to {summary_path}")
