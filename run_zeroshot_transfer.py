import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import tifffile as tiff
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from load_model import load_encoder

# Create directories
os.makedirs("results", exist_ok=True)
os.makedirs("images", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==========================================
# 1. Dataset Loader Functions
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
    
    batch_size = 4096
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):
            batch = torch.tensor(X_norm[i:i+batch_size], dtype=torch.float32, device=device)
            feats = encoder(batch)
            embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def align_raw_spectra(X, source_waves, target_waves):
    f = interp1d(source_waves, X, axis=1, bounds_error=False, fill_value=0.0)
    return f(target_waves).astype(np.float32)

# ==========================================
# 2. Semantic Mapping (All 220,985 Labeled Pixels)
# ==========================================

SEMANTIC_NAMES = {0: "Water", 1: "Trees/Vegetation", 2: "Soils", 3: "Urban"}

def get_semantic_mapping_all(name, y):
    sem_y = np.full_like(y, -1)
    if name == "Indian Pines":
        sem_y[y == 1] = 1 # Alfalfa -> Veg
        sem_y[y == 2] = 1 # Corn-notill -> Veg
        sem_y[y == 3] = 1 # Corn-mintill -> Veg
        sem_y[y == 4] = 1 # Corn -> Veg
        sem_y[y == 5] = 1 # Grass-pasture -> Veg
        sem_y[y == 6] = 1 # Grass-trees -> Veg
        sem_y[y == 7] = 1 # Grass-pasture-mowed -> Veg
        sem_y[y == 8] = 1 # Hay-windrowed -> Veg
        sem_y[y == 9] = 1 # Oats -> Veg
        sem_y[y == 10] = 1 # Soybean-notill -> Veg
        sem_y[y == 11] = 1 # Soybean-mintill -> Veg
        sem_y[y == 12] = 1 # Soybean-clean -> Veg
        sem_y[y == 13] = 1 # Wheat -> Veg
        sem_y[y == 14] = 1 # Woods -> Veg
        sem_y[y == 15] = 3 # Buildings-Grass-Trees-Drives -> Urban
        sem_y[y == 16] = 3 # Stone-Steel-Towers -> Urban
    elif name == "Botswana":
        sem_y[y == 1] = 0 # Water -> Water
        sem_y[y == 2] = 1 # Hippo grass -> Veg
        sem_y[y == 3] = 1 # Floodplain grasses 1 -> Veg
        sem_y[y == 4] = 1 # Floodplain grasses 2 -> Veg
        sem_y[y == 5] = 1 # Reeds -> Veg
        sem_y[y == 6] = 1 # Riparian -> Veg
        sem_y[y == 7] = 2 # Firescar -> Soils
        sem_y[y == 8] = 1 # Island interior -> Veg
        sem_y[y == 9] = 1 # Acacia woodlands -> Veg
        sem_y[y == 10] = 1 # Acacia shrublands -> Veg
        sem_y[y == 11] = 1 # Acacia grasslands -> Veg
        sem_y[y == 12] = 1 # Short mopane -> Veg
        sem_y[y == 13] = 1 # Mixed mopane -> Veg
        sem_y[y == 14] = 2 # Exposed soils -> Soils
    elif name == "Pavia University":
        sem_y[y == 1] = 3 # Asphalt -> Urban
        sem_y[y == 2] = 1 # Meadows -> Veg
        sem_y[y == 3] = 3 # Gravel -> Urban
        sem_y[y == 4] = 1 # Trees -> Veg
        sem_y[y == 5] = 3 # Painted metal sheets -> Urban
        sem_y[y == 6] = 2 # Bare Soil -> Soils
        sem_y[y == 7] = 3 # Bitumen -> Urban
        sem_y[y == 8] = 3 # Self-blocking bricks -> Urban
        sem_y[y == 9] = 3 # Shadows -> Urban
    elif name == "Pavia Centre":
        sem_y[y == 1] = 0 # Water -> Water
        sem_y[y == 2] = 1 # Trees -> Veg
        sem_y[y == 3] = 3 # Asphalt -> Urban
        sem_y[y == 4] = 3 # Self-blocking bricks -> Urban
        sem_y[y == 5] = 3 # Bitumen -> Urban
        sem_y[y == 6] = 3 # Tiles -> Urban
        sem_y[y == 7] = 3 # Shadows -> Urban
        sem_y[y == 8] = 1 # Meadows -> Veg
        sem_y[y == 9] = 2 # Bare Soil -> Soils
    elif name == "HyRank":
        sem_y[y == 1] = 3 # Dense Urban Fabric -> Urban
        sem_y[y == 2] = 3 # Mineral Extraction Sites -> Urban
        sem_y[y == 3] = 2 # Non-irrigated Arable Land -> Soils
        sem_y[y == 4] = 1 # Fruit Trees -> Veg
        sem_y[y == 5] = 1 # Olive Groves -> Veg
        sem_y[y == 6] = 1 # Coniferous Forest -> Veg
        sem_y[y == 7] = 1 # Deciduous Forest -> Veg
        sem_y[y == 8] = 1 # Mixed Forest -> Veg
        sem_y[y == 9] = 1 # Sparsely Vegetated Areas -> Veg
        sem_y[y == 10] = 0 # Water Courses -> Water
        sem_y[y == 11] = 0 # Water Bodies -> Water
        sem_y[y == 12] = 0 # Wetland -> Water
        sem_y[y == 13] = 0 # Salt Marshes -> Water
        sem_y[y == 14] = 1 # Natural Grassland -> Veg
    return sem_y

# ==========================================
# 3. Model Definition
# ==========================================

class PyTorchMLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes=4):
        super(PyTorchMLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def train_classifier_gpu(X_tr, y_tr, device="cuda", epochs=20, batch_size=1024):
    model = PyTorchMLPClassifier(X_tr.shape[1], num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    
    num_samples = X_t.shape[0]
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(num_samples, device=device)
        for i in range(0, num_samples, batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            bx, by = X_t[indices], y_t[indices]
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
    model.eval()
    return model

def predict_classifier_gpu(model, X_val, active_classes, device="cuda", batch_size=4096):
    model.eval()
    X_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            batch = X_t[i:i+batch_size]
            logits = model(batch)
            mask = torch.full((logits.shape[1],), -1e9, device=device)
            for c in active_classes:
                mask[c] = 0.0
            masked_logits = logits + mask
            pred_classes = torch.argmax(masked_logits, dim=1)
            preds.append(pred_classes.cpu().numpy())
    return np.concatenate(preds, axis=0)

# ==========================================
# 4. Pipeline Execution
# ==========================================

def main():
    datasets = ["Indian Pines", "Botswana", "Pavia University", "Pavia Centre", "HyRank"]
    data_dict = {}
    target_waves = np.linspace(400.0, 2500.0, 128)

    print("Loading all datasets and extracting features...")
    for name in datasets:
        print(f"Loading {name}...")
        if name == "Indian Pines":
            img, gt = load_indian_pines()
            waves = np.linspace(400.0, 2500.0, img.shape[2])
        elif name == "Botswana":
            img, gt = load_botswana()
            waves = np.linspace(400.0, 2500.0, img.shape[2])
        elif name == "Pavia University":
            img, gt = load_pavia_university()
            waves = np.linspace(430.0, 860.0, img.shape[2])
        elif name == "Pavia Centre":
            img, gt = load_pavia_centre()
            waves = np.linspace(430.0, 860.0, img.shape[2])
        elif name == "HyRank":
            img, gt = load_hyrank()
            with open("data/hyrank/hyrank_satellite.yaml") as f:
                hyrank_cfg = yaml.safe_load(f)
            waves = np.array(hyrank_cfg["info"]["wavelengths"]) * 1000.0
            
        H, W, C = img.shape
        X_flat = img.reshape(-1, C)
        y_flat = gt.reshape(-1)
        
        # Mapping all valid pixels
        sem_y = get_semantic_mapping_all(name, y_flat)
        valid_mask = sem_y >= 0
        
        X_valid = X_flat[valid_mask]
        y_valid = sem_y[valid_mask]
        X_norm = preprocess_hsi(X_valid)
        
        # Extract Embeddings
        print(f"Extracting embeddings for {name}...")
        if name in ["Pavia University", "Pavia Centre", "HyRank"]:
            X_emb = extract_embeddings(X_norm, band_dim=C, wavelengths=waves, device=DEVICE)
        else:
            X_emb = extract_embeddings(X_norm, band_dim=C, device=DEVICE)
            
        # Align Raw spectra
        print(f"Aligning Raw spectra for {name}...")
        X_raw_aligned = align_raw_spectra(X_norm, waves, target_waves)
        
        # Train/Test Split target dataset (for local evaluation of zero-shot)
        X_raw_tr, X_raw_te, X_emb_tr, X_emb_te, y_tr, y_te = train_test_split(
            X_raw_aligned, X_emb, y_valid, test_size=0.20, random_state=42, stratify=y_valid
        )
        
        data_dict[name] = {
            "raw_full": X_raw_aligned,
            "emb_full": X_emb,
            "y_full": y_valid,
            
            "raw_tr": X_raw_tr, "raw_te": X_raw_te,
            "emb_tr": X_emb_tr, "emb_te": X_emb_te,
            "y_tr": y_tr, "y_te": y_te
        }
        print(f"Dataset {name}: {len(y_valid)} mapped pixels.")

    print("\nRunning Zero-Shot Cross-Dataset Semantic Generalization Evaluation...")
    results_te = {}
    results_full = {}

    for target_name in datasets:
        print(f"\n--- Target: {target_name} (Zero-shot) ---")
        
        # Source data concatenation with domain-balanced sampling
        src_emb_list = []
        src_raw_list = []
        src_y_list = []
        for src_name in datasets:
            if src_name == target_name:
                continue
            
            # Sample at most 3000 pixels per source dataset to balance domains
            emb = data_dict[src_name]["emb_full"]
            raw = data_dict[src_name]["raw_full"]
            y = data_dict[src_name]["y_full"]
            
            np.random.seed(42)
            idx = np.arange(len(y))
            if len(y) > 3000:
                idx = np.random.choice(idx, 3000, replace=False)
                
            src_emb_list.append(emb[idx])
            src_raw_list.append(raw[idx])
            src_y_list.append(y[idx])
            
        X_src_emb = np.concatenate(src_emb_list, axis=0)
        X_src_raw = np.concatenate(src_raw_list, axis=0)
        y_src = np.concatenate(src_y_list, axis=0)
        
        # Target data
        X_tgt_raw_te = data_dict[target_name]["raw_te"]
        X_tgt_raw_full = data_dict[target_name]["raw_full"]
        X_tgt_emb_te = data_dict[target_name]["emb_te"]
        X_tgt_emb_full = data_dict[target_name]["emb_full"]
        y_tgt_te = data_dict[target_name]["y_te"]
        y_tgt_full = data_dict[target_name]["y_full"]
        
        active_classes = list(np.unique(y_tgt_full))
        active_class_names = [SEMANTIC_NAMES[c] for c in active_classes]
        print(f"Active semantic classes in target {target_name}: {active_class_names}")
        
        # Correct domain scaling: fit scaler on source data only, apply to target
        scaler_emb = StandardScaler()
        X_src_emb_s = scaler_emb.fit_transform(X_src_emb)
        X_tgt_emb_te_s = scaler_emb.transform(X_tgt_emb_te)
        X_tgt_emb_full_s = scaler_emb.transform(X_tgt_emb_full)
        
        scaler_raw = StandardScaler()
        X_src_raw_s = scaler_raw.fit_transform(X_src_raw)
        X_tgt_raw_te_s = scaler_raw.transform(X_tgt_raw_te)
        X_tgt_raw_full_s = scaler_raw.transform(X_tgt_raw_full)
        
        # Class balancing in source domain using oversampling
        unique_classes, class_counts = np.unique(y_src, return_counts=True)
        max_count = np.max(class_counts)
        
        X_src_raw_b = []
        X_src_emb_b = []
        y_src_b = []
        np.random.seed(42)
        for c in unique_classes:
            idx = np.where(y_src == c)[0]
            replicated_idx = np.random.choice(idx, size=max_count, replace=True)
            X_src_raw_b.append(X_src_raw_s[replicated_idx])
            X_src_emb_b.append(X_src_emb_s[replicated_idx])
            y_src_b.append(np.full(max_count, c))
            
        X_src_raw_s = np.concatenate(X_src_raw_b, axis=0)
        X_src_emb_s = np.concatenate(X_src_emb_b, axis=0)
        y_src_y = np.concatenate(y_src_b, axis=0)
        
        # 1. Train MLP on Raw
        print("Training PyTorch MLP on Raw spectra...")
        raw_model = train_classifier_gpu(X_src_raw_s, y_src_y, device=DEVICE, epochs=20, batch_size=1024)
        raw_preds_te = predict_classifier_gpu(raw_model, X_tgt_raw_te_s, active_classes, device=DEVICE)
        raw_f1_te = f1_score(y_tgt_te, raw_preds_te, average="macro")
        
        raw_preds_full = predict_classifier_gpu(raw_model, X_tgt_raw_full_s, active_classes, device=DEVICE)
        raw_f1_full = f1_score(y_tgt_full, raw_preds_full, average="macro")
        
        # 2. Train MLP on Embeddings
        print("Training PyTorch MLP on Embeddings...")
        emb_model = train_classifier_gpu(X_src_emb_s, y_src_y, device=DEVICE, epochs=20, batch_size=1024)
        emb_preds_te = predict_classifier_gpu(emb_model, X_tgt_emb_te_s, active_classes, device=DEVICE)
        emb_f1_te = f1_score(y_tgt_te, emb_preds_te, average="macro")
        
        emb_preds_full = predict_classifier_gpu(emb_model, X_tgt_emb_full_s, active_classes, device=DEVICE)
        emb_f1_full = f1_score(y_tgt_full, emb_preds_full, average="macro")
        
        print(f"Zero-shot Raw F1 (Test / Full): {raw_f1_te:.4f} / {raw_f1_full:.4f}")
        print(f"Zero-shot Emb F1 (Test / Full): {emb_f1_te:.4f} / {emb_f1_full:.4f}")
        
        results_te[target_name] = {"raw": raw_f1_te, "emb": emb_f1_te}
        results_full[target_name] = {"raw": raw_f1_full, "emb": emb_f1_full}

    # ==========================================
    # 5. Save Summary Table & Generate Visualization
    # ==========================================
    
    out_path = "results/zeroshot_metrics.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Zero-shot Cross-Dataset Semantic Generalization F1-Scores ===\n\n")
        f.write("| Dataset | Raw F1 (Test) | Emb F1 (Test) | Raw F1 (Full) | Emb F1 (Full) | Improvement (Δ Full) |\n")
        f.write("|---------|--------------|--------------|--------------|--------------|----------------------|\n")
        for name in datasets:
            te_raw = results_te[name]["raw"]
            te_emb = results_te[name]["emb"]
            full_raw = results_full[name]["raw"]
            full_emb = results_full[name]["emb"]
            diff = full_emb - full_raw
            f.write(f"| {name:<17} | {te_raw:.4f}       | {te_emb:.4f}       | {full_raw:.4f}       | {full_emb:.4f}       | {diff:+.4f}              |\n")
        
        # Average F1
        avg_te_raw = np.mean([results_te[n]["raw"] for n in datasets])
        avg_te_emb = np.mean([results_te[n]["emb"] for n in datasets])
        avg_full_raw = np.mean([results_full[n]["raw"] for n in datasets])
        avg_full_emb = np.mean([results_full[n]["emb"] for n in datasets])
        f.write(f"| {'Average':<17} | {avg_te_raw:.4f}       | {avg_te_emb:.4f}       | {avg_full_raw:.4f}       | {avg_full_emb:.4f}       | {avg_full_emb - avg_full_raw:+.4f}              |\n")

    print(f"\nZero-shot results saved to {out_path}")

    # Plot Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.35

    rects1 = ax.bar(x - width/2, [results_full[n]["raw"] for n in datasets], width, label='Raw Spectrums (Aligned)', color='#FFA07A', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x + width/2, [results_full[n]["emb"] for n in datasets], width, label='Hyperfocus Embeddings', color='#4682B4', edgecolor='black', alpha=0.9)

    ax.set_ylabel('Macro F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Zero-shot Cross-Dataset Semantic Generalization Performance (Full Dataset)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10, fontweight='bold')
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(fontsize=11)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plot_img_path = "images/zeroshot_performance.png"
    plt.savefig(plot_img_path, dpi=150)
    plt.close()
    print(f"Saved zero-shot plot to {plot_img_path}")

if __name__ == "__main__":
    main()
