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
from load_model import load_encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using GNN device: {DEVICE}")

# Create directories
os.makedirs("results", exist_ok=True)
os.makedirs("images", exist_ok=True)

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
    
    batch_size = 4096
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):
            batch = torch.tensor(X_norm[i:i+batch_size], dtype=torch.float32, device=device)
            feats = encoder(batch)
            embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def get_semantic_mapping_all(name, y):
    sem_y = np.full_like(y, -1)
    if name == "Indian Pines":
        sem_y[y == 1] = 1
        sem_y[y == 2] = 1
        sem_y[y == 3] = 1
        sem_y[y == 4] = 1
        sem_y[y == 5] = 1
        sem_y[y == 6] = 1
        sem_y[y == 7] = 1
        sem_y[y == 8] = 1
        sem_y[y == 9] = 1
        sem_y[y == 10] = 1
        sem_y[y == 11] = 1
        sem_y[y == 12] = 1
        sem_y[y == 13] = 1
        sem_y[y == 14] = 1
        sem_y[y == 15] = 3
        sem_y[y == 16] = 3
    elif name == "Botswana":
        sem_y[y == 1] = 0
        sem_y[y == 2] = 1
        sem_y[y == 3] = 1
        sem_y[y == 4] = 1
        sem_y[y == 5] = 1
        sem_y[y == 6] = 1
        sem_y[y == 7] = 2
        sem_y[y == 8] = 1
        sem_y[y == 9] = 1
        sem_y[y == 10] = 1
        sem_y[y == 11] = 1
        sem_y[y == 12] = 1
        sem_y[y == 13] = 1
        sem_y[y == 14] = 2
    elif name == "Pavia University":
        sem_y[y == 1] = 3
        sem_y[y == 2] = 1
        sem_y[y == 3] = 3
        sem_y[y == 4] = 1
        sem_y[y == 5] = 3
        sem_y[y == 6] = 2
        sem_y[y == 7] = 3
        sem_y[y == 8] = 3
        sem_y[y == 9] = 3
    elif name == "Pavia Centre":
        sem_y[y == 1] = 0
        sem_y[y == 2] = 1
        sem_y[y == 3] = 3
        sem_y[y == 4] = 3
        sem_y[y == 5] = 3
        sem_y[y == 6] = 3
        sem_y[y == 7] = 3
        sem_y[y == 8] = 1
        sem_y[y == 9] = 2
    elif name == "HyRank":
        sem_y[y == 1] = 3
        sem_y[y == 2] = 3
        sem_y[y == 3] = 2
        sem_y[y == 4] = 1
        sem_y[y == 5] = 1
        sem_y[y == 6] = 1
        sem_y[y == 7] = 1
        sem_y[y == 8] = 1
        sem_y[y == 9] = 1
        sem_y[y == 10] = 0
        sem_y[y == 11] = 0
        sem_y[y == 12] = 0
        sem_y[y == 13] = 0
        sem_y[y == 14] = 1
    return sem_y

# ==========================================
# 2. Fast Adjacency Graph Builder
# ==========================================

def build_spatial_adjacency(gt_shape, valid_coords):
    H, W = gt_shape
    N = len(valid_coords)
    grid_idx = np.full((H, W), -1, dtype=np.int32)
    for i, (r, c) in enumerate(valid_coords):
        grid_idx[r, c] = i
        
    edges_src = []
    edges_dst = []
    
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dr, dc in offsets:
        nr = valid_coords[:, 0] + dr
        nc = valid_coords[:, 1] + dc
        
        in_bounds = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        neighbor_nodes = np.full(N, -1, dtype=np.int32)
        neighbor_nodes[in_bounds] = grid_idx[nr[in_bounds], nc[in_bounds]]
        
        valid_edges = neighbor_nodes >= 0
        edges_src.extend(np.where(valid_edges)[0])
        edges_dst.extend(neighbor_nodes[valid_edges])
        
    edges_src = np.array(edges_src, dtype=np.int64)
    edges_dst = np.array(edges_dst, dtype=np.int64)
    return edges_src, edges_dst

# ==========================================
# 3. GNN Sparse Operator Helpers
# ==========================================

def get_gcn_norm_adj(edges_src, edges_dst, num_nodes, device="cuda"):
    src = torch.cat([torch.tensor(edges_src, device=device), torch.arange(num_nodes, device=device)])
    dst = torch.cat([torch.tensor(edges_dst, device=device), torch.arange(num_nodes, device=device)])
    
    deg = torch.zeros(num_nodes, device=device)
    deg = deg.scatter_add(0, src, torch.ones_like(src, dtype=torch.float32))
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    
    val = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    indices = torch.stack([src, dst])
    return torch.sparse_coo_tensor(indices, val, (num_nodes, num_nodes), device=device)

def get_sage_norm_adj(edges_src, edges_dst, num_nodes, device="cuda"):
    src = torch.tensor(edges_src, device=device)
    dst = torch.tensor(edges_dst, device=device)
    
    deg = torch.zeros(num_nodes, device=device)
    deg = deg.scatter_add(0, src, torch.ones_like(src, dtype=torch.float32))
    deg_inv = torch.pow(deg, -1.0)
    deg_inv[torch.isinf(deg_inv)] = 0.0
    
    val = deg_inv[src]
    indices = torch.stack([src, dst])
    return torch.sparse_coo_tensor(indices, val, (num_nodes, num_nodes), device=device)

# ==========================================
# 4. GNN Models
# ==========================================

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x, adj_norm):
        x = self.fc1(x)
        x = torch.sparse.mm(adj_norm, x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sparse.mm(adj_norm, x)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(GraphSAGE, self).__init__()
        self.fc1 = nn.Linear(in_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x, adj_mean):
        neigh1 = torch.sparse.mm(adj_mean, x)
        h1 = torch.cat([x, neigh1], dim=-1)
        h1 = self.relu(self.fc1(h1))
        h1 = self.dropout(h1)
        
        neigh2 = torch.sparse.mm(adj_mean, h1)
        h2 = torch.cat([h1, neigh2], dim=-1)
        out = self.fc2(h2)
        return out

# ==========================================
# 5. Training Loop
# ==========================================

def train_gnn(model, x, y, adj, train_mask, epochs=150, lr=0.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, adj)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        out = model(x, adj)
    return out.cpu().numpy()

# ==========================================
# 6. Main Pipeline
# ==========================================

def main():
    datasets = ["Indian Pines", "Botswana", "Pavia University", "Pavia Centre", "HyRank"]
    results = {}
    
    for name in datasets:
        print(f"\n=====================================")
        print(f"Processing Graph Classification for {name}...")
        print(f"=====================================")
        
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
        y_flat = gt.reshape(-1)
        sem_y = get_semantic_mapping_all(name, y_flat)
        valid_mask = sem_y >= 0
        
        # Nodes extraction
        valid_coords = np.argwhere(valid_mask.reshape(H, W))
        y_valid = sem_y[valid_mask]
        X_flat = img.reshape(-1, C)
        X_valid = X_flat[valid_mask]
        X_norm = preprocess_hsi(X_valid)
        
        # Extract Embeddings
        print(f"Extracting embeddings...")
        if name in ["Pavia University", "Pavia Centre", "HyRank"]:
            X_emb = extract_embeddings(X_norm, band_dim=C, wavelengths=waves, device=DEVICE)
        else:
            X_emb = extract_embeddings(X_norm, band_dim=C, device=DEVICE)
            
        # Standardize features
        scaler = StandardScaler()
        X_raw_s = scaler.fit_transform(X_norm)
        X_emb_s = scaler.fit_transform(X_emb)
        
        # Build Spatial Graph
        print(f"Building spatial 8-neighborhood graph...")
        src_edges, dst_edges = build_spatial_adjacency((H, W), valid_coords)
        num_nodes = len(y_valid)
        print(f"Graph nodes: {num_nodes:,} | Edges: {len(src_edges):,}")
        
        # Active classes and mapping to contiguous class IDs
        unique_classes = np.unique(y_valid)
        num_classes = len(unique_classes)
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        y_mapped = np.array([class_to_idx[c] for c in y_valid], dtype=np.int64)
        
        # 80/20 train/test split of node labels
        indices = np.arange(num_nodes)
        train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_mapped)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
        train_mask[train_idx] = True
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
        test_mask[test_idx] = True
        
        # Convert to GPU tensors
        x_raw_t = torch.tensor(X_raw_s, dtype=torch.float32, device=DEVICE)
        x_emb_t = torch.tensor(X_emb_s, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y_mapped, dtype=torch.long, device=DEVICE)
        
        # Build adjacency matrices
        adj_norm = get_gcn_norm_adj(src_edges, dst_edges, num_nodes, device=DEVICE)
        adj_mean = get_sage_norm_adj(src_edges, dst_edges, num_nodes, device=DEVICE)
        
        results[name] = {}
        
        # ==========================================
        # GCN Classification
        # ==========================================
        # GCN on Raw
        print("Training GCN on Raw spectra...")
        gcn_raw_model = GCN(in_dim=C, hidden_dim=128, num_classes=num_classes).to(DEVICE)
        out_gcn_raw = train_gnn(gcn_raw_model, x_raw_t, y_t, adj_norm, train_mask)
        pred_gcn_raw = np.argmax(out_gcn_raw, axis=1)
        f1_gcn_raw_te = f1_score(y_mapped[test_idx], pred_gcn_raw[test_idx], average="macro")
        f1_gcn_raw_all = f1_score(y_mapped, pred_gcn_raw, average="macro")
        
        # GCN on Embeddings
        print("Training GCN on Embeddings...")
        gcn_emb_model = GCN(in_dim=128, hidden_dim=128, num_classes=num_classes).to(DEVICE)
        out_gcn_emb = train_gnn(gcn_emb_model, x_emb_t, y_t, adj_norm, train_mask)
        pred_gcn_emb = np.argmax(out_gcn_emb, axis=1)
        f1_gcn_emb_te = f1_score(y_mapped[test_idx], pred_gcn_emb[test_idx], average="macro")
        f1_gcn_emb_all = f1_score(y_mapped, pred_gcn_emb, average="macro")
        
        # ==========================================
        # GraphSAGE Classification
        # ==========================================
        # GraphSAGE on Raw
        print("Training GraphSAGE on Raw spectra...")
        sage_raw_model = GraphSAGE(in_dim=C, hidden_dim=128, num_classes=num_classes).to(DEVICE)
        out_sage_raw = train_gnn(sage_raw_model, x_raw_t, y_t, adj_mean, train_mask)
        pred_sage_raw = np.argmax(out_sage_raw, axis=1)
        f1_sage_raw_te = f1_score(y_mapped[test_idx], pred_sage_raw[test_idx], average="macro")
        f1_sage_raw_all = f1_score(y_mapped, pred_sage_raw, average="macro")
        
        # GraphSAGE on Embeddings
        print("Training GraphSAGE on Embeddings...")
        sage_emb_model = GraphSAGE(in_dim=128, hidden_dim=128, num_classes=num_classes).to(DEVICE)
        out_sage_emb = train_gnn(sage_emb_model, x_emb_t, y_t, adj_mean, train_mask)
        pred_sage_emb = np.argmax(out_sage_emb, axis=1)
        f1_sage_emb_te = f1_score(y_mapped[test_idx], pred_sage_emb[test_idx], average="macro")
        f1_sage_emb_all = f1_score(y_mapped, pred_sage_emb, average="macro")
        
        results[name] = {
            "gcn_raw_te": f1_gcn_raw_te, "gcn_raw_all": f1_gcn_raw_all,
            "gcn_emb_te": f1_gcn_emb_te, "gcn_emb_all": f1_gcn_emb_all,
            "sage_raw_te": f1_sage_raw_te, "sage_raw_all": f1_sage_raw_all,
            "sage_emb_te": f1_sage_emb_te, "sage_emb_all": f1_sage_emb_all
        }
        
        print(f"GCN Raw F1 (Test/All): {f1_gcn_raw_te:.4f} / {f1_gcn_raw_all:.4f}")
        print(f"GCN Emb F1 (Test/All): {f1_gcn_emb_te:.4f} / {f1_gcn_emb_all:.4f}")
        print(f"SAGE Raw F1 (Test/All): {f1_sage_raw_te:.4f} / {f1_sage_raw_all:.4f}")
        print(f"SAGE Emb F1 (Test/All): {f1_sage_emb_te:.4f} / {f1_sage_emb_all:.4f}")

    # ==========================================
    # 7. Write GNN Metrics
    # ==========================================
    
    gnn_out_path = "results/gnn_metrics.txt"
    with open(gnn_out_path, "w", encoding="utf-8") as f:
        f.write("=== Spatial Graph Classification (GCN vs GraphSAGE) F1-Scores ===\n\n")
        f.write("| Dataset | Model | Raw F1 (Test) | Emb F1 (Test) | Raw F1 (Full) | Emb F1 (Full) | Improvement (Full) |\n")
        f.write("|---------|-------|--------------|--------------|--------------|--------------|--------------------|\n")
        for name in datasets:
            res = results[name]
            f.write(f"| {name:<17} | GCN   | {res['gcn_raw_te']:.4f}       | {res['gcn_emb_te']:.4f}       | {res['gcn_raw_all']:.4f}       | {res['gcn_emb_all']:.4f}       | {res['gcn_emb_all'] - res['gcn_raw_all']:+.4f}            |\n")
            f.write(f"| {name:<17} | SAGE  | {res['sage_raw_te']:.4f}       | {res['sage_emb_te']:.4f}       | {res['sage_raw_all']:.4f}       | {res['sage_emb_all']:.4f}       | {res['sage_emb_all'] - res['sage_raw_all']:+.4f}            |\n")
        
        # Averages
        avg_gcn_raw_te = np.mean([results[n]["gcn_raw_te"] for n in datasets])
        avg_gcn_emb_te = np.mean([results[n]["gcn_emb_te"] for n in datasets])
        avg_gcn_raw_all = np.mean([results[n]["gcn_raw_all"] for n in datasets])
        avg_gcn_emb_all = np.mean([results[n]["gcn_emb_all"] for n in datasets])
        
        avg_sage_raw_te = np.mean([results[n]["sage_raw_te"] for n in datasets])
        avg_sage_emb_te = np.mean([results[n]["sage_emb_te"] for n in datasets])
        avg_sage_raw_all = np.mean([results[n]["sage_raw_all"] for n in datasets])
        avg_sage_emb_all = np.mean([results[n]["sage_emb_all"] for n in datasets])
        
        f.write(f"| {'Average':<17} | GCN   | {avg_gcn_raw_te:.4f}       | {avg_gcn_emb_te:.4f}       | {avg_gcn_raw_all:.4f}       | {avg_gcn_emb_all:.4f}       | {avg_gcn_emb_all - avg_gcn_raw_all:+.4f}            |\n")
        f.write(f"| {'Average':<17} | SAGE  | {avg_sage_raw_te:.4f}       | {avg_sage_emb_te:.4f}       | {avg_sage_raw_all:.4f}       | {avg_sage_emb_all:.4f}       | {avg_sage_emb_all - avg_sage_raw_all:+.4f}            |\n")

    print(f"\nGNN performance metrics saved to {gnn_out_path}")

    # ==========================================
    # 8. Generate Visualizations
    # ==========================================
    
    # 8.1 GNN performance bar chart
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(datasets))
    width = 0.35
    
    # GCN Panel
    axes[0].bar(x - width/2, [results[n]["gcn_raw_all"] for n in datasets], width, label='Raw features', color='#FFA07A', edgecolor='black', alpha=0.9)
    axes[0].bar(x + width/2, [results[n]["gcn_emb_all"] for n in datasets], width, label='Hyperfocus Emb', color='#4682B4', edgecolor='black', alpha=0.9)
    axes[0].set_ylabel('Macro F1-Score', fontsize=11, fontweight='bold')
    axes[0].set_title('GCN Spatial Classification Performance (Full Nodes)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, fontsize=9, rotation=15)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    axes[0].legend()
    
    # GraphSAGE Panel
    axes[1].bar(x - width/2, [results[n]["sage_raw_all"] for n in datasets], width, label='Raw features', color='#FFA07A', edgecolor='black', alpha=0.9)
    axes[1].bar(x + width/2, [results[n]["sage_emb_all"] for n in datasets], width, label='Hyperfocus Emb', color='#4682B4', edgecolor='black', alpha=0.9)
    axes[1].set_ylabel('Macro F1-Score', fontsize=11, fontweight='bold')
    axes[1].set_title('GraphSAGE Spatial Classification Performance (Full Nodes)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, fontsize=9, rotation=15)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    axes[1].legend()
    
    plt.tight_layout()
    gnn_plot_path = "images/gnn_performance.png"
    plt.savefig(gnn_plot_path, dpi=150)
    plt.close()
    print(f"Saved GNN performance bar chart to {gnn_plot_path}")
    
    # 8.2 Generate GNN Architecture Schematic diagram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Drawing boxes
    bbox_props = dict(boxstyle="round,pad=0.3", fc="lightblue", ec="b", lw=2)
    bbox_props_emb = dict(boxstyle="round,pad=0.3", fc="#e6f2ff", ec="#0066cc", lw=2)
    bbox_props_raw = dict(boxstyle="round,pad=0.3", fc="#ffe6e6", ec="#cc0000", lw=2)
    
    # Title
    ax.text(0.5, 0.95, "Spatial Graph Neural Network (GNN/GCN) Architecture Flow", 
            ha="center", va="center", fontsize=14, fontweight="bold", bbox=dict(boxstyle="square", fc="w", ec="gray", alpha=0.9))
    
    # Grid Pixels (Nodes)
    ax.text(0.15, 0.70, "1. HSI Grid Pixels\n(Node coordinates: x, y)\nActive pixels = Nodes", ha="center", va="center", bbox=bbox_props, fontsize=9)
    
    # Spatial Graph
    ax.text(0.50, 0.70, "2. Spatial Graph Construction\n(Connect each active pixel to\nits spatial 8-neighbors)", ha="center", va="center", bbox=bbox_props, fontsize=9)
    
    # Features
    ax.text(0.15, 0.40, "3a. Raw Spectra Features\n[N, 128] Aligned Bands", ha="center", va="center", bbox=bbox_props_raw, fontsize=9)
    ax.text(0.15, 0.15, "3b. Hyperfocus Embeddings\n[N, 128] Latent Vectors", ha="center", va="center", bbox=bbox_props_emb, fontsize=9)
    
    # Aggregation
    ax.text(0.50, 0.28, "4. GNN Message Passing Layer\n\n- GCN:  H^(l+1) = ReLU(D^(-1/2) A_tilde D^(-1/2) H^l W^l)\n- SAGE: H^(l+1) = ReLU( [ H^l , Mean_Neigh(H^l) ] W^l )", 
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", lw=2), fontsize=9)
    
    # Output Classification
    ax.text(0.85, 0.28, "5. Node Prediction\n- Transductive Train (80%)\n- Evaluation on Test (20%)\n- Output: F1 Class Score", ha="center", va="center", bbox=bbox_props, fontsize=9)
    
    # Arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="gray")
    
    ax.annotate("", xy=(0.33, 0.70), xytext=(0.28, 0.70), arrowprops=arrow_props)
    ax.annotate("", xy=(0.50, 0.38), xytext=(0.50, 0.62), arrowprops=arrow_props)
    ax.annotate("", xy=(0.33, 0.28), xytext=(0.28, 0.40), arrowprops=arrow_props)
    ax.annotate("", xy=(0.33, 0.28), xytext=(0.28, 0.15), arrowprops=arrow_props)
    ax.annotate("", xy=(0.74, 0.28), xytext=(0.67, 0.28), arrowprops=arrow_props)
    
    plt.tight_layout()
    arch_plot_path = "images/gnn_architecture.png"
    plt.savefig(arch_plot_path, dpi=150)
    plt.close()
    print(f"Saved GNN architecture schematic to {arch_plot_path}")

if __name__ == "__main__":
    main()
