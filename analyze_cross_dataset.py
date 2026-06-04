import os
import yaml
import torch
import numpy as np
import scipy.io as sio
import tifffile as tiff
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from load_model import load_encoder

# Create directories
os.makedirs("images/cross_dataset", exist_ok=True)
os.makedirs("results", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==========================================
# 1. Dataset Loader Functions
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

# ==========================================
# 3. Geometric / Statistical Functions
# ==========================================

def draw_confidence_ellipse(x, y, ax, color, n_std=1.2, **kwargs):
    if len(x) < 5:
        return None, None, None
    cov = np.cov(x, y)
    std_x = np.sqrt(cov[0, 0])
    std_y = np.sqrt(cov[1, 1])
    if std_x < 1e-6 or std_y < 1e-6:
        return None, None, None
        
    pearson = cov[0, 1] / (std_x * std_y)
    pearson = np.clip(pearson, -0.999, 0.999)
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor='none', edgecolor=color, linewidth=1.2, alpha=0.55, linestyle='--', **kwargs)
    
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
    return (mean_x, mean_y), cov, (scale_x, scale_y)

def calculate_mahalanobis(x, y, mean, cov):
    """
    Computes the square of Mahalanobis distance for 2D points.
    """
    diff = np.column_stack((x - mean[0], y - mean[1]))
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.eye(2) * 1e6
    dist_sq = np.sum(diff @ inv_cov * diff, axis=1)
    return dist_sq

# ==========================================
# 4. Preparing Baseline & Intruder Data
# ==========================================

print("Loading all datasets and extracting embeddings...")

# 1. Indian Pines
img_ip, gt_ip = load_indian_pines()
X_ip_flat = img_ip.reshape(-1, img_ip.shape[2])
y_ip_flat = gt_ip.reshape(-1)
valid_idx_ip = np.where(y_ip_flat > 0)[0]
X_ip_valid = X_ip_flat[valid_idx_ip]
y_ip_valid = y_ip_flat[valid_idx_ip]
X_ip_norm = preprocess_hsi(X_ip_valid)
X_ip_emb = extract_embeddings(X_ip_norm, band_dim=img_ip.shape[2], device=DEVICE)
X_ip_emb = StandardScaler().fit_transform(X_ip_emb)

# 2. Botswana
img_bo, gt_bo = load_botswana()
X_bo_flat = img_bo.reshape(-1, img_bo.shape[2])
y_bo_flat = gt_bo.reshape(-1)
valid_idx_bo = np.where(y_bo_flat > 0)[0]
X_bo_valid = X_bo_flat[valid_idx_bo]
X_bo_norm = preprocess_hsi(X_bo_valid)
X_bo_emb = extract_embeddings(X_bo_norm, band_dim=img_bo.shape[2], device=DEVICE)
X_bo_emb = StandardScaler().fit_transform(X_bo_emb)

# 3. Pavia University
img_pu, gt_pu = load_pavia_university()
X_pu_flat = img_pu.reshape(-1, img_pu.shape[2])
y_pu_flat = gt_pu.reshape(-1)
valid_idx_pu = np.where(y_pu_flat > 0)[0]
X_pu_valid = X_pu_flat[valid_idx_pu]
X_pu_norm = preprocess_hsi(X_pu_valid)
pu_waves = np.linspace(430.0, 860.0, img_pu.shape[2])
X_pu_emb = extract_embeddings(X_pu_norm, band_dim=img_pu.shape[2], wavelengths=pu_waves, device=DEVICE)
X_pu_emb = StandardScaler().fit_transform(X_pu_emb)

# 4. Pavia Centre
img_pc, gt_pc = load_pavia_centre()
X_pc_flat = img_pc.reshape(-1, img_pc.shape[2])
y_pc_flat = gt_pc.reshape(-1)
valid_idx_pc = np.where(y_pc_flat > 0)[0]
X_pc_valid = X_pc_flat[valid_idx_pc]
X_pc_norm = preprocess_hsi(X_pc_valid)
pc_waves = np.linspace(430.0, 860.0, img_pc.shape[2])
X_pc_emb = extract_embeddings(X_pc_norm, band_dim=img_pc.shape[2], wavelengths=pc_waves, device=DEVICE)
X_pc_emb = StandardScaler().fit_transform(X_pc_emb)

# 5. HyRank
img_hr, gt_hr = load_hyrank()
X_hr_flat = img_hr.reshape(-1, img_hr.shape[2])
y_hr_flat = gt_hr.reshape(-1)
valid_idx_hr = np.where(y_hr_flat > 0)[0]
X_hr_valid = X_hr_flat[valid_idx_hr]
X_hr_norm = preprocess_hsi(X_hr_valid)
with open("data/hyrank/hyrank_satellite.yaml") as f:
    hyrank_cfg = yaml.safe_load(f)
hr_waves = np.array(hyrank_cfg["info"]["wavelengths"]) * 1000.0
X_hr_emb = extract_embeddings(X_hr_norm, band_dim=img_hr.shape[2], wavelengths=hr_waves, device=DEVICE)
X_hr_emb = StandardScaler().fit_transform(X_hr_emb)

print("All embeddings extracted and standard scaled.")

# Stratified Sampling to speed up t-SNE and prevent overplotting
def stratified_sample(X, y, target_n=3000, seed=42):
    np.random.seed(seed)
    unique_y, counts = np.unique(y, return_counts=True)
    sampled_indices = []
    for val, count in zip(unique_y, counts):
        idx = np.where(y == val)[0]
        n = min(len(idx), max(30, int(count * (target_n / len(y)))))
        sampled_indices.extend(np.random.choice(idx, n, replace=False))
    return np.array(sampled_indices)

def random_sample(X, target_n=3000, seed=42):
    np.random.seed(seed)
    if len(X) <= target_n:
        return np.arange(len(X))
    return np.random.choice(len(X), target_n, replace=False)

# Sample datasets
idx_ip = stratified_sample(X_ip_norm, y_ip_valid, target_n=3000)
X_raw_ip = X_ip_norm[idx_ip]
X_emb_ip = X_ip_emb[idx_ip]
y_ip = y_ip_valid[idx_ip]

idx_bo = random_sample(X_bo_norm, target_n=1500)
X_raw_bo = X_bo_norm[idx_bo]
X_emb_bo = X_bo_emb[idx_bo]

idx_pu = random_sample(X_pu_norm, target_n=2000)
X_raw_pu = X_pu_norm[idx_pu]
X_emb_pu = X_pu_emb[idx_pu]

idx_pc = random_sample(X_pc_norm, target_n=2000)
X_raw_pc = X_pc_norm[idx_pc]
X_emb_pc = X_pc_emb[idx_pc]

idx_hr = random_sample(X_hr_norm, target_n=2000)
X_raw_hr = X_hr_norm[idx_hr]
X_emb_hr = X_hr_emb[idx_hr]

datasets = {
    "Botswana": {"raw": X_raw_bo, "emb": X_emb_bo, "color": "indigo", "label": "Botswana"},
    "Pavia University": {"raw": X_raw_pu, "emb": X_emb_pu, "color": "darkorange", "label": "Pavia Univ"},
    "Pavia Centre": {"raw": X_raw_pc, "emb": X_emb_pc, "color": "grey", "label": "Pavia Centre"},
    "HyRank": {"raw": X_raw_hr, "emb": X_emb_hr, "color": "teal", "label": "HyRank"}
}

# ==========================================
# 5. Analysis Pipeline
# ==========================================

IP_CLASS_NAMES = {
    1: "Alfalfa", 2: "Corn-notill", 3: "Corn-mintill", 4: "Corn",
    5: "Grass-pasture", 6: "Grass-trees", 7: "Grass-pasture-mowed",
    8: "Hay-windrowed", 9: "Oats", 10: "Soybean-notill",
    11: "Soybean-mintill", 12: "Soybean-clean", 13: "Wheat",
    14: "Woods", 15: "Buildings-Grass-Trees-Drives", 16: "Stone-Steel-Towers"
}

def analyze_case(case_num, case_name, intruder_keys):
    print(f"\n--- Running Case {case_num}: {case_name} ---")
    
    # Collect intruder data
    if intruder_keys:
        X_raw_int_list = [datasets[k]["raw"] for k in intruder_keys]
        X_emb_int_list = [datasets[k]["emb"] for k in intruder_keys]
        
        # We need to match features if band counts differ
        # raw features: pad with zeros to match maximum bands
        max_bands = max(X_raw_ip.shape[1], *(x.shape[1] for x in X_raw_int_list))
        
        def pad_channels(X, target_dim):
            if X.shape[1] == target_dim:
                return X
            padded = np.zeros((X.shape[0], target_dim))
            padded[:, :X.shape[1]] = X
            return padded
            
        X_raw_ip_padded = pad_channels(X_raw_ip, max_bands)
        X_raw_int_padded_list = [pad_channels(x, max_bands) for x in X_raw_int_list]
        
        X_raw_intruders = np.concatenate(X_raw_int_padded_list, axis=0)
        X_emb_intruders = np.concatenate(X_emb_int_list, axis=0)
    else:
        max_bands = X_raw_ip.shape[1]
        X_raw_ip_padded = X_raw_ip
        X_raw_intruders = np.empty((0, max_bands))
        X_emb_intruders = np.empty((0, X_emb_ip.shape[1]))
        
    # --- 1. Raw Spectrum PCA ---
    pca_raw = PCA(n_components=2, random_state=42)
    X_raw_ip_pca = pca_raw.fit_transform(X_raw_ip_padded)
    X_raw_int_pca = pca_raw.transform(X_raw_intruders) if len(X_raw_intruders) > 0 else np.empty((0, 2))
    
    # --- 2. Embedding PCA ---
    pca_emb = PCA(n_components=2, random_state=42)
    X_emb_ip_pca = pca_emb.fit_transform(X_emb_ip)
    X_emb_int_pca = pca_emb.transform(X_emb_intruders) if len(X_emb_intruders) > 0 else np.empty((0, 2))
    
    # --- 3. Raw Spectrum t-SNE (StandardScaler + LDA fit on IP, then joint t-SNE) ---
    scaler_raw_lda = StandardScaler()
    X_raw_ip_scaled = scaler_raw_lda.fit_transform(X_raw_ip_padded)
    X_raw_int_scaled = scaler_raw_lda.transform(X_raw_intruders) if len(X_raw_intruders) > 0 else np.empty((0, max_bands))
    
    lda_raw = LinearDiscriminantAnalysis(n_components=min(15, len(np.unique(y_ip)) - 1))
    X_raw_ip_lda = lda_raw.fit_transform(X_raw_ip_scaled, y_ip)
    X_raw_int_lda = lda_raw.transform(X_raw_int_scaled) if len(X_raw_intruders) > 0 else np.empty((0, X_raw_ip_lda.shape[1]))
    
    # Joint t-SNE
    if len(X_raw_intruders) > 0:
        X_raw_combined_lda = np.concatenate([X_raw_ip_lda, X_raw_int_lda], axis=0)
    else:
        X_raw_combined_lda = X_raw_ip_lda
        
    tsne_raw = TSNE(n_components=2, perplexity=45, random_state=42, max_iter=1000, init='pca')
    X_raw_combined_tsne = tsne_raw.fit_transform(X_raw_combined_lda)
    
    X_raw_ip_tsne = X_raw_combined_tsne[:len(X_raw_ip)]
    X_raw_int_tsne = X_raw_combined_tsne[len(X_raw_ip):] if len(X_raw_intruders) > 0 else np.empty((0, 2))
    
    # --- 4. Embedding t-SNE (StandardScaler + LDA fit on IP, then joint t-SNE) ---
    scaler_emb_lda = StandardScaler()
    X_emb_ip_scaled = scaler_emb_lda.fit_transform(X_emb_ip)
    X_emb_int_scaled = scaler_emb_lda.transform(X_emb_intruders) if len(X_emb_intruders) > 0 else np.empty((0, X_emb_ip.shape[1]))
    
    lda_emb = LinearDiscriminantAnalysis(n_components=min(15, len(np.unique(y_ip)) - 1))
    X_emb_ip_lda = lda_emb.fit_transform(X_emb_ip_scaled, y_ip)
    X_emb_int_lda = lda_emb.transform(X_emb_int_scaled) if len(X_emb_intruders) > 0 else np.empty((0, X_emb_ip_lda.shape[1]))
    
    # Joint t-SNE
    if len(X_emb_intruders) > 0:
        X_emb_combined_lda = np.concatenate([X_emb_ip_lda, X_emb_int_lda], axis=0)
    else:
        X_emb_combined_lda = X_emb_ip_lda
        
    tsne_emb = TSNE(n_components=2, perplexity=45, random_state=42, max_iter=1000, init='pca')
    X_emb_combined_tsne = tsne_emb.fit_transform(X_emb_combined_lda)
    
    X_emb_ip_tsne = X_emb_combined_tsne[:len(X_emb_ip)]
    X_emb_int_tsne = X_emb_combined_tsne[len(X_emb_ip):] if len(X_emb_intruders) > 0 else np.empty((0, 2))
    
    # --- 5. Quantitative Metric Calculations ---
    # Metric 1: Dataset Discriminability (KNN Classifier accuracy on IP vs. Intruders)
    knn_raw_acc, knn_emb_acc = 0.0, 0.0
    if len(X_raw_intruders) > 0:
        y_combined = np.zeros(len(X_raw_ip) + len(X_raw_intruders))
        y_combined[len(X_raw_ip):] = 1 # Intruder is 1
        
        # Raw Discriminability in the full dimension (padded to max_bands)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        X_raw_combined = np.concatenate([X_raw_ip_padded, X_raw_intruders], axis=0)
        raw_scores = []
        for train_idx, test_idx in skf.split(X_raw_combined, y_combined):
            knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
            knn.fit(X_raw_combined[train_idx], y_combined[train_idx])
            raw_scores.append(knn.score(X_raw_combined[test_idx], y_combined[test_idx]))
        knn_raw_acc = np.mean(raw_scores)
        
        # Embedding Discriminability
        X_emb_combined = np.concatenate([X_emb_ip, X_emb_intruders], axis=0)
        emb_scores = []
        for train_idx, test_idx in skf.split(X_emb_combined, y_combined):
            knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
            knn.fit(X_emb_combined[train_idx], y_combined[train_idx])
            emb_scores.append(knn.score(X_emb_combined[test_idx], y_combined[test_idx]))
        knn_emb_acc = np.mean(emb_scores)
        
    # Metric 2: Intrusion Rate in 2D Spaces (PCA & t-SNE)
    # We will compute the percentage of intruder points falling inside the 1.2 std ellipses of IP classes.
    def compute_intrusion_rate(X_ip_2d, X_int_2d):
        if len(X_int_2d) == 0:
            return 0.0
        inside_count = 0
        total_valid_ellipses = 0
        
        for cls in np.unique(y_ip):
            cls_mask = (y_ip == cls)
            x_c = X_ip_2d[cls_mask, 0]
            y_c = X_ip_2d[cls_mask, 1]
            if len(x_c) < 5:
                continue
            cov = np.cov(x_c, y_c)
            mean = (np.mean(x_c), np.mean(y_c))
            if cov[0, 0] < 1e-6 or cov[1, 1] < 1e-6:
                continue
            
            # Mahalanobis distances for all intruder points relative to this IP class ellipse
            dist_sq = calculate_mahalanobis(X_int_2d[:, 0], X_int_2d[:, 1], mean, cov)
            # 1.2 std threshold is equivalent to d_M^2 <= 1.2^2
            inside_mask = (dist_sq <= 1.2**2)
            inside_count += np.sum(inside_mask)
            total_valid_ellipses += 1
            
        # Overall Intrusion Rate is the sum of intruder points entering any IP class ellipse divided by the intruder count
        # Or relative to total points inside. Let's define it as: (Intruder points inside any ellipse) / (Total intruder points)
        # However, because ellipses can overlap, we take the union.
        if total_valid_ellipses == 0:
            return 0.0
            
        union_inside = np.zeros(len(X_int_2d), dtype=bool)
        for cls in np.unique(y_ip):
            cls_mask = (y_ip == cls)
            x_c = X_ip_2d[cls_mask, 0]
            y_c = X_ip_2d[cls_mask, 1]
            if len(x_c) < 5:
                continue
            cov = np.cov(x_c, y_c)
            mean = (np.mean(x_c), np.mean(y_c))
            if cov[0, 0] < 1e-6 or cov[1, 1] < 1e-6:
                continue
            dist_sq = calculate_mahalanobis(X_int_2d[:, 0], X_int_2d[:, 1], mean, cov)
            union_inside |= (dist_sq <= 1.2**2)
            
        return np.mean(union_inside) * 100.0

    raw_pca_intrusion = compute_intrusion_rate(X_raw_ip_pca, X_raw_int_pca)
    emb_pca_intrusion = compute_intrusion_rate(X_emb_ip_pca, X_emb_int_pca)
    raw_tsne_intrusion = compute_intrusion_rate(X_raw_ip_tsne, X_raw_int_tsne)
    emb_tsne_intrusion = compute_intrusion_rate(X_emb_ip_tsne, X_emb_int_tsne)
    
    # --- 6. Visualization plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    cmap = plt.get_cmap("tab20")
    
    # Helper to plot a scatter view
    def plot_view(ax, X_ip_2d, X_int_2d, title):
        # 1. Plot Intruders first (background layer)
        if len(X_int_2d) > 0:
            start_idx = 0
            for key in intruder_keys:
                n_pts = len(datasets[key]["raw"])
                pts = X_int_2d[start_idx : start_idx + n_pts]
                ax.scatter(pts[:, 0], pts[:, 1], color=datasets[key]["color"], s=6, alpha=0.30, 
                           label=datasets[key]["label"], zorder=1)
                start_idx += n_pts
                
        # 2. Plot Indian Pines classes on top
        for cls in sorted(np.unique(y_ip)):
            cls_mask = (y_ip == cls)
            color = cmap((cls - 1) % 20)
            ax.scatter(X_ip_2d[cls_mask, 0], X_ip_2d[cls_mask, 1], color=color, s=14, alpha=0.75, 
                       label=f"IP Class {cls}" if len(intruder_keys) == 0 else None, zorder=2)
            draw_confidence_ellipse(X_ip_2d[cls_mask, 0], X_ip_2d[cls_mask, 1], ax, color=color, n_std=1.2, zorder=3)
            
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
    plot_view(axes[0, 0], X_raw_ip_pca, X_raw_int_pca, f"Raw Spectrum PCA\n(Intrusion: {raw_pca_intrusion:.2f}%)")
    plot_view(axes[0, 1], X_emb_ip_pca, X_emb_int_pca, f"Hyperfocus Embedding PCA\n(Intrusion: {emb_pca_intrusion:.2f}%)")
    plot_view(axes[1, 0], X_raw_ip_tsne, X_raw_int_tsne, f"Raw Spectrum t-SNE (Joint)\n(Intrusion: {raw_tsne_intrusion:.2f}%)")
    plot_view(axes[1, 1], X_emb_ip_tsne, X_emb_int_tsne, f"Hyperfocus Embedding t-SNE (Joint)\n(Intrusion: {emb_tsne_intrusion:.2f}%)")
    
    # Setup single consolidated legend
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
                
    # Add Indian Pines class descriptions if space permits
    if not intruder_keys:
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5), title="Indian Pines Classes", fontsize=8)
    else:
        # Just show the main datasets in the legend to avoid cluttering
        filtered_handles = [h for h, l in zip(handles, labels) if not l.startswith("IP Class")]
        filtered_labels = [l for l in labels if not l.startswith("IP Class")]
        # Add a dummy handle for Indian Pines
        ip_handle = plt.Line2D([0], [0], marker='o', color='royalblue', linestyle='', markersize=8)
        filtered_handles.insert(0, ip_handle)
        filtered_labels.insert(0, "Indian Pines (All Crops)")
        
        fig.legend(filtered_handles, filtered_labels, loc='center right', bbox_to_anchor=(0.99, 0.5), title="Datasets", fontsize=10)
        
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    
    img_save_path = f"images/cross_dataset/case{case_num}_{case_name.lower().replace(' ', '_')}.png"
    plt.savefig(img_save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {img_save_path}")
    
    return {
        "case_num": case_num,
        "case_name": case_name,
        "raw_pca_intrusion": raw_pca_intrusion,
        "emb_pca_intrusion": emb_pca_intrusion,
        "raw_tsne_intrusion": raw_tsne_intrusion,
        "emb_tsne_intrusion": emb_tsne_intrusion,
        "knn_raw_acc": knn_raw_acc,
        "knn_emb_acc": knn_emb_acc,
        "image_path": img_save_path
    }

# ==========================================
# 6. Execute All Analytical Cases
# ==========================================

cases = [
    (0, "Indian Pines Baseline", []),
    (1, "IP + Botswana", ["Botswana"]),
    (2, "IP + Pavia University", ["Pavia University"]),
    (3, "IP + Pavia Centre", ["Pavia Centre"]),
    (4, "IP + HyRank", ["HyRank"]),
    (5, "IP + Botswana + Pavia University", ["Botswana", "Pavia University"]),
    (6, "IP + Botswana + Pavia Centre", ["Botswana", "Pavia Centre"]),
    (7, "IP + Botswana + HyRank", ["Botswana", "HyRank"]),
    (8, "IP + Pavia University + Pavia Centre", ["Pavia University", "Pavia Centre"]),
    (9, "IP + Pavia University + HyRank", ["Pavia University", "HyRank"]),
    (10, "IP + Botswana + Pavia University + Pavia Centre", ["Botswana", "Pavia University", "Pavia Centre"]),
    (11, "IP + All Datasets", ["Botswana", "Pavia University", "Pavia Centre", "HyRank"])
]

case_results = []
for c_num, c_name, intruders in cases:
    try:
        res = analyze_case(c_num, c_name, intruders)
        case_results.append(res)
    except Exception as e:
        print(f"Failed executing Case {c_num} ({c_name}): {e}")
        import traceback
        traceback.print_exc()

# Save final text report summary
txt_summary_path = "results/cross_dataset_summary.txt"
with open(txt_summary_path, "w", encoding="utf-8") as f:
    f.write("=== Cross-Dataset HSI Embedding Interference Summary ===\n\n")
    f.write(f"| Case | {'Case Name':<42} | {'Raw PCA Intr.':<14} | {'Emb PCA Intr.':<14} | {'Raw tSNE Intr.':<14} | {'Emb tSNE Intr.':<14} | {'Raw KNN Acc.':<12} | {'Emb KNN Acc.':<12} |\n")
    f.write(f"|{'--'*3}|{'-'*44}|{'-'*16}|{'-'*16}|{'-'*16}|{'-'*16}|{'-'*14}|{'-'*14}|\n")
    for r in case_results:
        f.write(f"| {r['case_num']:<4} | {r['case_name']:<42} | {r['raw_pca_intrusion']:<14.2f}% | {r['emb_pca_intrusion']:<14.2f}% | {r['raw_tsne_intrusion']:<14.2f}% | {r['emb_tsne_intrusion']:<14.2f}% | {r['knn_raw_acc']:<12.4f} | {r['knn_emb_acc']:<12.4f} |\n")

print(f"\nAll cases completed successfully! Summary saved to {txt_summary_path}")
