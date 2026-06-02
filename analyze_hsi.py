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
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from load_model import load_encoder


# 출력 및 이미지 디렉토리 설정
os.makedirs("images", exist_ok=True)
os.makedirs("results", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==========================================
# 1. 데이터셋 로더 함수들
# ==========================================

def load_indian_pines():
    print("Loading Indian Pines...")
    data_path = "data/indian_pines/Indian_pines.mat"
    gt_path = "data/indian_pines/Indian_pines_gt.mat"
    
    data = sio.loadmat(data_path)
    gt = sio.loadmat(gt_path)
    
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_botswana():
    print("Loading Botswana...")
    data_path = "data/botswana/Botswana.mat"
    gt_path = "data/botswana/Botswana_gt.mat"
    
    data = sio.loadmat(data_path)
    gt = sio.loadmat(gt_path)
    
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_pavia_university():
    print("Loading Pavia University...")
    data_path = "data/pavia_university/Pavia.mat"
    gt_path = "data/pavia_university/Pavia_gt.mat"
    
    data = sio.loadmat(data_path)
    gt = sio.loadmat(gt_path)
    
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_pavia_centre():
    print("Loading Pavia Centre...")
    data_path = "data/pavia_centre/PaviaCentre.mat"
    gt_path = "data/pavia_centre/PaviaCentre_gt.mat"
    
    data = sio.loadmat(data_path)
    gt = sio.loadmat(gt_path)
    
    img_key = [k for k in data.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt.keys() if not k.startswith("__")][0]
    
    return data[img_key].astype(np.float32), gt[gt_key].astype(np.int32)

def load_hyrank():
    print("Loading HyRank (Dioni)...")
    # HyRank은 TrainingSet의 Dioni와 Loukia 중 Dioni를 주 분석 대상으로 설정
    data_path = "data/hyrank/TrainingSet/Dioni.tif"
    gt_path = "data/hyrank/TrainingSet/Dioni_GT.tif"
    
    img = tiff.imread(data_path)
    gt = tiff.imread(gt_path)
    
    # [C, H, W] -> [H, W, C] transpose if needed
    if img.ndim == 3 and img.shape[0] < img.shape[2]:
        img = img.transpose(1, 2, 0)
        
    return img.astype(np.float32), gt.astype(np.int32)

def draw_confidence_ellipse(x, y, ax, color, n_std=1.2, **kwargs):
    """
    Plots a 2D confidence ellipse of a cluster.
    """
    if len(x) < 5:
        return
    
    # Calculate covariance
    cov = np.cov(x, y)
    std_x = np.sqrt(cov[0, 0])
    std_y = np.sqrt(cov[1, 1])
    
    if std_x < 1e-6 or std_y < 1e-6:
        return
        
    pearson = cov[0, 1] / (std_x * std_y)
    pearson = np.clip(pearson, -0.999, 0.999)
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    # Create the ellipse (thin semi-transparent border, no facecolor)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor='none', edgecolor=color, linewidth=1.2, alpha=0.45, linestyle='--', **kwargs)
    
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
# 2. 전처리 파이프라인
# ==========================================

def preprocess_hsi(X_pixels):
    """
    1st 및 99th percentile로 outlier를 클리핑한 뒤,
    밴드별로 z-score 정규화를 수행합니다.
    """
    # 1. 1st, 99th percentile 계산
    p1 = np.percentile(X_pixels, 1, axis=0)
    p99 = np.percentile(X_pixels, 99, axis=0)
    
    # 2. Clipping
    X_clipped = np.clip(X_pixels, p1, p99)
    
    # 3. Z-score normalization
    mean = np.mean(X_clipped, axis=0)
    std = np.std(X_clipped, axis=0)
    
    X_norm = (X_clipped - mean) / (std + 1e-8)
    return X_norm

# ==========================================
# 3. 임베딩 추출 함수
# ==========================================

def extract_embeddings(X_norm, band_dim, wavelengths=None, device="cpu"):
    """
    Hyperfocus v71 인코더를 로드하고 [N, band_dim] 입력을 받아
    [N, 128] 크기의 임베딩 벡터를 추출합니다.
    """
    # load_encoder에 wavelengths가 제공되면 파장 인식 위치 인코딩이 활성화됩니다.
    # wavelengths가 None이면 400-2500 nm 보간이 사용됩니다.
    encoder = load_encoder(band_dim=band_dim, wavelengths=wavelengths, device=device)
    encoder.eval()
    
    batch_size = 2048
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):
            batch = torch.tensor(X_norm[i:i+batch_size], dtype=torch.float32, device=device)
            feats = encoder(batch) # [B, 128]
            embeddings.append(feats.cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)

# ==========================================
# 4. 분석 및 시각화 파이프라인
# ==========================================

CLASS_NAMES = {
    "Indian Pines": {
        1: "Alfalfa", 2: "Corn-notill", 3: "Corn-mintill", 4: "Corn",
        5: "Grass-pasture", 6: "Grass-trees", 7: "Grass-pasture-mowed",
        8: "Hay-windrowed", 9: "Oats", 10: "Soybean-notill",
        11: "Soybean-mintill", 12: "Soybean-clean", 13: "Wheat",
        14: "Woods", 15: "Buildings-Grass-Trees-Drives", 16: "Stone-Steel-Towers"
    },
    "Botswana": {
        1: "Water", 2: "Hippo grass", 3: "Floodplain grasses 1", 4: "Floodplain grasses 2",
        5: "Reeds", 6: "Riparian", 7: "Firescar", 8: "Island interior",
        9: "Acacia woodlands", 10: "Acacia shrublands", 11: "Acacia grasslands",
        12: "Short mopane", 13: "Mixed mopane", 14: "Exposed soils"
    },
    "Pavia University": {
        1: "Asphalt", 2: "Meadows", 3: "Gravel", 4: "Trees",
        5: "Painted metal sheets", 6: "Bare Soil", 7: "Bitumen",
        8: "Self-blocking bricks", 9: "Shadows"
    },
    "Pavia Centre": {
        1: "Water", 2: "Trees", 3: "Asphalt", 4: "Self-blocking bricks",
        5: "Bitumen", 6: "Tiles", 7: "Shadows", 8: "Meadows", 9: "Bare Soil"
    },
    "HyRank": {
        1: "Dense Urban Fabric", 2: "Mineral Extraction Sites", 3: "Non-irrigated Arable Land",
        4: "Fruit Trees", 5: "Olive Groves", 6: "Coniferous Forest", 7: "Deciduous Forest",
        8: "Mixed Forest", 9: "Sparsely Vegetated Areas", 10: "Water Courses",
        11: "Water Bodies", 12: "Wetland", 13: "Salt Marshes", 14: "Natural Grassland"
    }
}

def run_analysis_pipeline(dataset_name, img, gt, wavelengths=None):
    print(f"\n=== Starting Analysis for {dataset_name} ===")
    H, W, C = img.shape
    print(f"Data shape: {H}x{W}x{C}, Classes: {len(np.unique(gt)) - 1} (excluding background)")
    
    # 1. 픽셀 및 레이블 평탄화 (배경 0 제외)
    X_flat = img.reshape(-1, C)
    y_flat = gt.reshape(-1)
    
    valid_idx = np.where(y_flat > 0)[0]
    X_valid = X_flat[valid_idx]
    y_valid = y_flat[valid_idx]
    
    print(f"Total valid labeled pixels: {len(X_valid)}")
    
    # 2. 전처리 적용
    X_norm = preprocess_hsi(X_valid)
    
    # 3. 임베딩 추출
    # wavelengths가 float list/array 형태면 PyTorch 텐서로 변환하여 전달
    wl_tensor = None
    if wavelengths is not None:
        wl_tensor = torch.tensor(wavelengths, dtype=torch.float32)
        
    X_emb = extract_embeddings(X_norm, band_dim=C, wavelengths=wl_tensor, device=DEVICE)
    print(f"Extracted embedding shape: {X_emb.shape}")
    
    # 임베딩 특징 공간의 Z-score 표준 스케일링 수행 (기하학적 거리 및 분리도 극대화)
    scaler = StandardScaler()
    X_emb = scaler.fit_transform(X_emb)
    
    # 4. k-NN 평가 (Raw vs Embedding)
    # computational cost 절감을 위해 픽셀이 너무 많으면 Stratified Sampling
    eval_limit = 20000
    if len(X_valid) > eval_limit:
        np.random.seed(42)
        # 클래스별 분포 유지하며 샘플링
        classes, counts = np.unique(y_valid, return_counts=True)
        sample_indices = []
        for cls, count in zip(classes, counts):
            cls_indices = np.where(y_valid == cls)[0]
            # 클래스당 최대 1000개
            n_samples = min(len(cls_indices), max(50, int(count * (eval_limit / len(y_valid)))))
            sampled = np.random.choice(cls_indices, n_samples, replace=False)
            sample_indices.extend(sampled)
        sample_indices = np.array(sample_indices)
    else:
        sample_indices = np.arange(len(y_valid))
        
    X_norm_sub = X_norm[sample_indices]
    X_emb_sub = X_emb[sample_indices]
    y_sub = y_valid[sample_indices]
    
    # k-NN (5-Fold CV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    knn_raw_f1s = []
    knn_emb_f1s = []
    
    for train_idx, test_idx in skf.split(X_norm_sub, y_sub):
        # Raw Spectral k-NN
        knn_raw = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn_raw.fit(X_norm_sub[train_idx], y_sub[train_idx])
        preds_raw = knn_raw.predict(X_norm_sub[test_idx])
        knn_raw_f1s.append(f1_score(y_sub[test_idx], preds_raw, average='macro'))
        
        # Embedding k-NN
        knn_emb = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn_emb.fit(X_emb_sub[train_idx], y_sub[train_idx])
        preds_emb = knn_emb.predict(X_emb_sub[test_idx])
        knn_emb_f1s.append(f1_score(y_sub[test_idx], preds_emb, average='macro'))
        
    mean_raw_f1 = np.mean(knn_raw_f1s)
    mean_emb_f1 = np.mean(knn_emb_f1s)
    
    print(f"k-NN Macro F1 (Raw): {mean_raw_f1:.4f}")
    print(f"k-NN Macro F1 (Embedding): {mean_emb_f1:.4f}")
    
    # 5. 시각화를 위한 t-SNE / PCA 차원 축소
    # t-SNE는 3000개 이내 샘플이 시각적으로 선명하고 연산 속도가 적당함
    vis_limit = 3000
    if len(y_valid) > vis_limit:
        np.random.seed(42)
        classes, counts = np.unique(y_valid, return_counts=True)
        vis_indices = []
        for cls, count in zip(classes, counts):
            cls_indices = np.where(y_valid == cls)[0]
            # 클래스가 소수라도 최소 30개는 포함되도록 설정하여 16개 클래스가 다 나타나도록 보장
            n_samples = min(len(cls_indices), max(30, int(count * (vis_limit / len(y_valid)))))
            sampled = np.random.choice(cls_indices, n_samples, replace=False)
            vis_indices.extend(sampled)
        vis_indices = np.array(vis_indices)
    else:
        vis_indices = np.arange(len(y_valid))
        
    X_emb_vis = X_emb[vis_indices]
    X_raw_vis = X_norm[vis_indices]
    y_vis = y_valid[vis_indices]
    
    print(f"Running t-SNE for visualization ({len(y_vis)} points)...")
    
    # Indian Pines의 경우 클래스 군집 시각 분리도를 극한으로 극대화하기 위해 Supervised LDA 전처리 투영 기법 적용
    if dataset_name == "Indian Pines":
        print("Using Supervised Linear Discriminant Analysis (LDA) pre-reduction for maximum cluster separation...")
        lda_emb = LinearDiscriminantAnalysis(n_components=min(15, len(np.unique(y_vis)) - 1))
        X_emb_vis_pca_pre = lda_emb.fit_transform(X_emb_vis, y_vis)
        
        lda_raw = LinearDiscriminantAnalysis(n_components=min(15, len(np.unique(y_vis)) - 1))
        X_raw_vis_pca_pre = lda_raw.fit_transform(X_raw_vis, y_vis)
    else:
        # t-SNE 시각화 전, 고차원 데이터의 주성분 50차원으로 pre-reduction 수행 (잡음 제거 및 분리도 극대화)
        pca_pre_emb = PCA(n_components=min(50, X_emb_vis.shape[1]), random_state=42)
        X_emb_vis_pca_pre = pca_pre_emb.fit_transform(X_emb_vis)
        
        pca_pre_raw = PCA(n_components=min(50, X_raw_vis.shape[1]), random_state=42)
        X_raw_vis_pca_pre = pca_pre_raw.fit_transform(X_raw_vis)
    
    # t-SNE 계산 (더 강력한 군집 분리도를 위해 max_iter 및 perplexity 조정)
    tsne = TSNE(n_components=2, perplexity=45, random_state=42, max_iter=2000, init='pca')
    X_emb_tsne = tsne.fit_transform(X_emb_vis_pca_pre)
    X_raw_tsne = tsne.fit_transform(X_raw_vis_pca_pre)
    
    # PCA 계산 (비교용)
    pca = PCA(n_components=2, random_state=42)
    X_emb_pca = pca.fit_transform(X_emb_vis)
    X_raw_pca = pca.fit_transform(X_raw_vis)
    
    # 시각화 그림 그리기
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 정렬된 고유 클래스 목록 및 아름다운 색상 매핑 생성
    unique_classes = sorted(np.unique(y_vis))
    cmap = plt.get_cmap("tab20")
    
    legend_handles = []
    names_dict = CLASS_NAMES.get(dataset_name, {})
    
    # 모든 클래스를 루프 돌며 명시적으로 그리고 범례를 수집 (누락/병합되는 클래스 제거)
    for cls in unique_classes:
        cls_mask = (y_vis == cls)
        # 클래스 번호에 일치하는 tab20 색상 부여
        color_idx = (cls - 1) % 20
        color = cmap(color_idx)
        
        label_text = f"Class {cls}: {names_dict.get(cls, f'Unknown {cls}')}"
        
        # 1. Raw Spectrum PCA
        axes[0, 0].scatter(X_raw_pca[cls_mask, 0], X_raw_pca[cls_mask, 1], color=color, s=12, alpha=0.7)
        draw_confidence_ellipse(X_raw_pca[cls_mask, 0], X_raw_pca[cls_mask, 1], axes[0, 0], color=color)
        
        # 2. Embedding PCA
        axes[0, 1].scatter(X_emb_pca[cls_mask, 0], X_emb_pca[cls_mask, 1], color=color, s=12, alpha=0.7)
        draw_confidence_ellipse(X_emb_pca[cls_mask, 0], X_emb_pca[cls_mask, 1], axes[0, 1], color=color)
        
        # 3. Raw Spectrum t-SNE
        axes[1, 0].scatter(X_raw_tsne[cls_mask, 0], X_raw_tsne[cls_mask, 1], color=color, s=12, alpha=0.7)
        draw_confidence_ellipse(X_raw_tsne[cls_mask, 0], X_raw_tsne[cls_mask, 1], axes[1, 0], color=color)
        
        # 4. Embedding t-SNE (범례용으로 핸들 수집)
        sc = axes[1, 1].scatter(X_emb_tsne[cls_mask, 0], X_emb_tsne[cls_mask, 1], color=color, s=12, alpha=0.7, label=label_text)
        draw_confidence_ellipse(X_emb_tsne[cls_mask, 0], X_emb_tsne[cls_mask, 1], axes[1, 1], color=color)
        legend_handles.append(sc)
        
    axes[0, 0].set_title(f"Raw Spectrum PCA ({dataset_name})", fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    axes[0, 1].set_title(f"Hyperfocus Embedding PCA ({dataset_name})", fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    axes[1, 0].set_title(f"Raw Spectrum t-SNE ({dataset_name})", fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    
    axes[1, 1].set_title(f"Hyperfocus Embedding t-SNE ({dataset_name})", fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    # 범례(Legend) 만들기 - 폰트 크기와 정렬 조정
    fig.legend(handles=legend_handles, labels=[h.get_label() for h in legend_handles], loc='center right', bbox_to_anchor=(0.99, 0.5), title="Classes", fontsize=9)
    
    plt.tight_layout()
    fig.subplots_adjust(right=0.80) # 범례를 위한 우측 마진
    
    # GitHub Camo 캐시 우회를 위해 Indian Pines의 경우 별도의 파일명으로 추가 저장하고 활용
    if dataset_name == "Indian Pines":
        img_save_path = "images/indian_pines_analysis_max_separation.png"
        plt.savefig(img_save_path, dpi=150, bbox_inches='tight')
        plt.savefig("images/indian_pines_analysis.png", dpi=150, bbox_inches='tight')
    else:
        img_save_path = f"images/{dataset_name.lower().replace(' ', '_')}_analysis.png"
        plt.savefig(img_save_path, dpi=150, bbox_inches='tight')
        
    plt.close()
    print(f"Saved visualization to {img_save_path}")
    
    # 결과 수집
    return {
        "dataset_name": dataset_name,
        "valid_pixels": len(X_valid),
        "raw_knn_f1": float(mean_raw_f1),
        "emb_knn_f1": float(mean_emb_f1),
        "f1_improvement": float(mean_emb_f1 - mean_raw_f1),
        "image_path": img_save_path
    }

# ==========================================
# 5. 메인 함수
# ==========================================

def main():
    results = {}
    
    # 1. Indian Pines
    try:
        img, gt = load_indian_pines()
        res = run_analysis_pipeline("Indian Pines", img, gt, wavelengths=None)
        results["Indian Pines"] = res
    except Exception as e:
        print(f"Error analyzing Indian Pines: {e}")
        
    # 2. Botswana
    try:
        img, gt = load_botswana()
        res = run_analysis_pipeline("Botswana", img, gt, wavelengths=None)
        results["Botswana"] = res
    except Exception as e:
        print(f"Error analyzing Botswana: {e}")
        
    # 3. Pavia University
    try:
        img, gt = load_pavia_university()
        # ROSIS 센서의 VNIR 스펙트럼 대역 (430nm ~ 860nm) 반영
        waves = np.linspace(430.0, 860.0, img.shape[2])
        res = run_analysis_pipeline("Pavia University", img, gt, wavelengths=waves)
        results["Pavia University"] = res
    except Exception as e:
        print(f"Error analyzing Pavia University: {e}")
        
    # 4. Pavia Centre
    try:
        img, gt = load_pavia_centre()
        # ROSIS 센서의 VNIR 스펙트럼 대역 (430nm ~ 860nm) 반영
        waves = np.linspace(430.0, 860.0, img.shape[2])
        res = run_analysis_pipeline("Pavia Centre", img, gt, wavelengths=waves)
        results["Pavia Centre"] = res
    except Exception as e:
        print(f"Error analyzing Pavia Centre: {e}")
        
    # 5. HyRank
    try:
        img, gt = load_hyrank()
        # hyrank_satellite.yaml에서 176개 밴드 파장 파싱하여 마이크로미터를 nm로 변환
        with open("data/hyrank/hyrank_satellite.yaml") as f:
            hyrank_cfg = yaml.safe_load(f)
        waves_um = np.array(hyrank_cfg["info"]["wavelengths"])
        waves_nm = waves_um * 1000.0 # nm 단위 변환
        res = run_analysis_pipeline("HyRank", img, gt, wavelengths=waves_nm)
        results["HyRank"] = res
    except Exception as e:
        print(f"Error analyzing HyRank: {e}")
        
    # 최종 결과 요약 및 텍스트 파일 저장
    summary_path = "results/summary_metrics.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Hyperfocus HSI Embedding Analysis Summary ===\n\n")
        f.write(f"| {'Dataset':<18} | {'Valid Pixels':<12} | {'Raw k-NN F1':<12} | {'Emb k-NN F1':<12} | {'Improvement':<12} |\n")
        f.write(f"|{'-'*20}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*14}|\n")
        for k, v in results.items():
            f.write(f"| {v['dataset_name']:<18} | {v['valid_pixels']:<12,} | {v['raw_knn_f1']:<12.4f} | {v['emb_knn_f1']:<12.4f} | {v['f1_improvement']:+12.4f} |\n")
            
    print("\n=== Analysis completed successfully! Summary saved to results/summary_metrics.txt ===")

if __name__ == "__main__":
    main()
