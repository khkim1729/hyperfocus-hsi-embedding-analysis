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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
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

# ==========================================
# 2. Load and extract embeddings for all
# ==========================================

print("Loading datasets and extracting embeddings...")

datasets_data = {}

# 1. Indian Pines
img, gt = load_indian_pines()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
mask = y_flat > 0
# Extract all labeled pixels (or sample for speed & balance)
np.random.seed(42)
idx = np.where(mask)[0]
sampled_idx = np.random.choice(idx, min(4000, len(idx)), replace=False)
X_ip = X_flat[sampled_idx]
y_ip = y_flat[sampled_idx]
X_ip_norm = preprocess_hsi(X_ip)
X_ip_emb = extract_embeddings(X_ip_norm, img.shape[2], device=DEVICE)
datasets_data["Indian Pines"] = {"raw": X_ip_norm, "emb": X_ip_emb, "y": y_ip}

# 2. Botswana
img, gt = load_botswana()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
mask = y_flat > 0
idx = np.where(mask)[0]
sampled_idx = np.random.choice(idx, min(2500, len(idx)), replace=False)
X_bo = X_flat[sampled_idx]
y_bo = y_flat[sampled_idx]
X_bo_norm = preprocess_hsi(X_bo)
X_bo_emb = extract_embeddings(X_bo_norm, img.shape[2], device=DEVICE)
datasets_data["Botswana"] = {"raw": X_bo_norm, "emb": X_bo_emb, "y": y_bo}

# 3. Pavia University
img, gt = load_pavia_university()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
mask = y_flat > 0
idx = np.where(mask)[0]
sampled_idx = np.random.choice(idx, min(4000, len(idx)), replace=False)
X_pu = X_flat[sampled_idx]
y_pu = y_flat[sampled_idx]
X_pu_norm = preprocess_hsi(X_pu)
pu_waves = np.linspace(430.0, 860.0, img.shape[2])
X_pu_emb = extract_embeddings(X_pu_norm, img.shape[2], pu_waves, device=DEVICE)
datasets_data["Pavia University"] = {"raw": X_pu_norm, "emb": X_pu_emb, "y": y_pu}

# 4. Pavia Centre
img, gt = load_pavia_centre()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
mask = y_flat > 0
idx = np.where(mask)[0]
sampled_idx = np.random.choice(idx, min(4000, len(idx)), replace=False)
X_pc = X_flat[sampled_idx]
y_pc = y_flat[sampled_idx]
X_pc_norm = preprocess_hsi(X_pc)
pc_waves = np.linspace(430.0, 860.0, img.shape[2])
X_pc_emb = extract_embeddings(X_pc_norm, img.shape[2], pc_waves, device=DEVICE)
datasets_data["Pavia Centre"] = {"raw": X_pc_norm, "emb": X_pc_emb, "y": y_pc}

# 5. HyRank
img, gt = load_hyrank()
X_flat = img.reshape(-1, img.shape[2])
y_flat = gt.reshape(-1)
mask = y_flat > 0
idx = np.where(mask)[0]
sampled_idx = np.random.choice(idx, min(4000, len(idx)), replace=False)
X_hr = X_flat[sampled_idx]
y_hr = y_flat[sampled_idx]
X_hr_norm = preprocess_hsi(X_hr)
with open("data/hyrank/hyrank_satellite.yaml") as f:
    hyrank_cfg = yaml.safe_load(f)
hr_waves = np.array(hyrank_cfg["info"]["wavelengths"]) * 1000.0
X_hr_emb = extract_embeddings(X_hr_norm, img.shape[2], hr_waves, device=DEVICE)
datasets_data["HyRank"] = {"raw": X_hr_norm, "emb": X_hr_emb, "y": y_hr}

# ==========================================
# 3. Fit Optimal Latent Space Projection (LDA on IP)
# ==========================================

print("Fitting optimal latent space projection model on Indian Pines...")

# Extract IP embeddings
X_ip_emb = datasets_data["Indian Pines"]["emb"]
y_ip = datasets_data["Indian Pines"]["y"]

# Fit Scaler and LDA
scaler = StandardScaler()
X_ip_emb_scaled = scaler.fit_transform(X_ip_emb)

# Number of classes is 16, so max n_components is 15
lda = LDA(n_components=15)
X_ip_latent = lda.fit_transform(X_ip_emb_scaled, y_ip)

print(f"Optimal Latent Space trained. Output shape: {X_ip_latent.shape}")

# ==========================================
# 4. Project and Evaluate Other Datasets
# ==========================================

print("Projecting and evaluating all datasets in the Optimal Latent Space...")

evaluation_results = {}
projected_coords_tsne = {}

for name, data in datasets_data.items():
    X_raw = data["raw"]
    X_emb = data["emb"]
    y = data["y"]
    
    # 1. Project embedding into Optimal Latent Space
    X_emb_scaled = scaler.transform(X_emb)
    # Project to 15D LDA space
    X_latent = lda.transform(X_emb_scaled)
    
    # 2. Evaluate Class Discriminability (KNN 5-Fold accuracy)
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Raw Accuracy (Using padded raw for dimension uniformity)
    X_raw_padded = pad_features(X_raw)
    raw_acc = np.mean(cross_val_score(knn, X_raw_padded, y, cv=5))
    
    # Embedding Latent Accuracy
    latent_acc = np.mean(cross_val_score(knn, X_latent, y, cv=5))
    
    # 3. Calculate Silhouette scores in 15D space
    raw_sil = silhouette_score(X_raw_padded, y)
    latent_sil = silhouette_score(X_latent, y)
    
    evaluation_results[name] = {
        "raw_knn": raw_acc,
        "latent_knn": latent_acc,
        "raw_sil": raw_sil,
        "latent_sil": latent_sil
    }
    
    # 4. Map to 2D using t-SNE for plotting
    print(f" - Running t-SNE for {name} (Optimal Latent Space)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca')
    coords_2d = tsne.fit_transform(X_latent)
    projected_coords_tsne[name] = coords_2d

# ==========================================
# 5. Visualizing Generalization
# ==========================================

print("Plotting generalization mappings...")

# 2x3 Plot showing all 5 datasets in the Optimal Latent Space
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Cross-Dataset Class Separation in the Indian Pines-trained Optimal Latent Space", fontsize=15, fontweight='bold')

# Flatten axes and remove the last one (since we only have 5 datasets)
axes_flat = axes.flatten()
fig.delaxes(axes_flat[-1])

for idx, (name, coords) in enumerate(projected_coords_tsne.items()):
    ax = axes_flat[idx]
    y = datasets_data[name]["y"]
    unique_classes = np.unique(y)
    
    # Use colormap to get distinct colors for each dataset's classes
    cmap = plt.get_cmap("tab20")
    for c_idx, c in enumerate(unique_classes):
        mask = (y == c)
        ax.scatter(coords[mask, 0], coords[mask, 1], color=cmap(c_idx / len(unique_classes)), s=10, alpha=0.55, label=f"Class {c}")
        
    ax.set_title(f"{name}\n(KNN Acc: {evaluation_results[name]['latent_knn']*100:.2f}%, Sil: {evaluation_results[name]['latent_sil']:.3f})", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("images/cross_dataset/optimal_latent_space_transfer_all.png", dpi=140, bbox_inches='tight')
plt.close()

# ==========================================
# 6. Save Results Summary
# ==========================================

report_path = "results/optimal_latent_space_summary.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=== Optimal Latent Space Generalization & Transferability ===\n\n")
    f.write("| Dataset | Raw KNN Acc | Latent KNN Acc | Raw Silhouette | Latent Silhouette |\n")
    f.write("|---------|-------------|----------------|----------------|-------------------|\n")
    for name, res in evaluation_results.items():
        f.write(f"| {name:<17} | {res['raw_knn']*100:.2f}%      | {res['latent_knn']*100:.2f}%         | {res['raw_sil']:.4f}         | {res['latent_sil']:.4f}            |\n")

print(f"\nOptimal Latent Space analysis completed successfully! Metrics saved to {report_path}")
