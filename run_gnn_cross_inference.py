import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import tifffile as tiff
from sklearn.preprocessing import StandardScaler
from run_gnn_classification import (
    load_indian_pines, load_botswana, load_pavia_university, load_hyrank,
    preprocess_hsi, extract_embeddings, build_spatial_adjacency, get_gcn_norm_adj,
    GCN
)
from run_gnn_classification_real_classes import CLASS_NAMES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using GNN Cross-Inference Device: {DEVICE}")

def load_target_data(name):
    print(f"Loading target dataset: {name}...")
    if name == "Indian Pines":
        img, gt = load_indian_pines()
        waves = np.linspace(400.0, 2500.0, img.shape[2])
    elif name == "Botswana":
        img, gt = load_botswana()
        waves = np.linspace(400.0, 2500.0, img.shape[2])
    elif name == "Pavia University":
        img, gt = load_pavia_university()
        waves = np.linspace(430.0, 860.0, img.shape[2])
    elif name == "HyRank":
        img, gt = load_hyrank()
        with open("data/hyrank/hyrank_satellite.yaml") as f:
            hyrank_cfg = yaml.safe_load(f)
        waves = np.array(hyrank_cfg["info"]["wavelengths"]) * 1000.0
    else:
        raise ValueError(f"Unknown dataset: {name}")

    H, W, C = img.shape
    y_flat = gt.reshape(-1)
    
    # Filter: select ALL non-background pixels (label > 0)
    valid_mask = y_flat > 0
    valid_coords = np.argwhere(valid_mask.reshape(H, W))
    y_valid = y_flat[valid_mask]
    
    X_flat = img.reshape(-1, C)
    X_valid = X_flat[valid_mask]
    X_norm = preprocess_hsi(X_valid)
    
    # Extract Embeddings
    if name in ["Pavia University", "HyRank"]:
        X_emb = extract_embeddings(X_norm, band_dim=C, wavelengths=waves, device=DEVICE)
    else:
        X_emb = extract_embeddings(X_norm, band_dim=C, device=DEVICE)
        
    return X_emb, y_valid, valid_coords, (H, W)

def run_cross_inference(source_name, target_name, X_emb_target, y_target, valid_coords, gt_shape):
    ckpt_path = f"checkpoints/gnn_real_classes/{source_name.lower().replace(' ', '_')}_gcn_emb.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run training first.")
        
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    
    # Reconstruct GCN Emb Model
    num_classes = checkpoint["num_classes"]
    model = GCN(in_dim=128, hidden_dim=128, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load Scaler from Source Checkpoint
    scaler_mean = checkpoint["scaler_mean"]
    scaler_scale = checkpoint["scaler_scale"]
    class_to_idx_src = checkpoint["class_to_idx"]
    unique_classes_src = checkpoint["unique_classes"]
    
    # Standardize target embeddings using source domain scaler
    X_emb_target_s = (X_emb_target - scaler_mean) / scaler_scale
    
    # Reconstruct Target Graph
    src_edges, dst_edges = build_spatial_adjacency(gt_shape, valid_coords)
    num_nodes = len(y_target)
    adj_gcn = get_gcn_norm_adj(src_edges, dst_edges, num_nodes, device=DEVICE)
    
    # Convert target features to tensor
    x_emb_t = torch.tensor(X_emb_target_s, dtype=torch.float32, device=DEVICE)
    
    # Run Inference
    with torch.no_grad():
        logits = model(x_emb_t, adj_gcn)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
    idx_to_class_src = {v: k for k, v in class_to_idx_src.items()}
    
    # For each target class, compute class prediction distribution
    target_unique_classes = np.unique(y_target)
    results = {}
    
    for t_cls in target_unique_classes:
        t_cls_name = CLASS_NAMES.get(target_name, {}).get(t_cls, f"Class {t_cls}")
        mask = y_target == t_cls
        cls_probs = probs[mask] # [num_nodes_in_cls, num_classes_src]
        
        # Average probability distribution for this target class
        avg_probs = np.mean(cls_probs, axis=0)
        
        # Get top 3 predicted classes
        top3_indices = np.argsort(avg_probs)[::-1][:3]
        top3_probs = avg_probs[top3_indices]
        
        top3_predictions = []
        for idx, prob in zip(top3_indices, top3_probs):
            src_cls_id = idx_to_class_src[idx]
            src_cls_name = CLASS_NAMES.get(source_name, {}).get(src_cls_id, f"Class {src_cls_id}")
            top3_predictions.append({
                "class_id": int(src_cls_id),
                "class_name": src_cls_name,
                "prob": float(prob)
            })
            
        results[t_cls] = {
            "target_class_id": int(t_cls),
            "target_class_name": t_cls_name,
            "top3": top3_predictions
        }
        
    return results

def main():
    sources = ["Indian Pines", "Pavia University", "HyRank"]
    targets = ["Indian Pines", "Botswana", "Pavia University", "HyRank"]
    
    # Cache extracted target embeddings to avoid redundant extraction
    data_cache = {}
    for t_name in targets:
        X_emb, y_valid, valid_coords, shape = load_target_data(t_name)
        data_cache[t_name] = (X_emb, y_valid, valid_coords, shape)
        
    output_md = []
    output_md.append("# 교차 데이터셋 GNN 모델 제로샷 추론 분석 보고서\n")
    output_md.append("본 보고서는 특정 초분광 데이터셋의 원본 클래스(Real Classes)로 학습된 GNN 모델을 사용하여 타겟 데이터셋에 직접 제로샷 추론(Zero-shot Inference)을 수행한 결과 분석 리포트입니다.\n")
    output_md.append("기초 모델 **Hyperfocus v71**이 추출한 128차원 임베딩을 노드 피처로 사용하고, 타겟 데이터셋의 공간 그래프 구조 상에서 인접 메시지를 전파하여 예측을 수행했습니다.\n")
    output_md.append("---\n")
    
    for s_name in sources:
        output_md.append(f"## {s_name} GNN 모델 추론 결과 분석\n")
        output_md.append(f"**{s_name}** 데이터셋(소스 도메인)으로 학습된 GCN Embedding 모델을 사용하여 타겟 데이터셋들에 추론을 수행했습니다. 소스 도메인의 StandardScaler 평균/표준편차로 타겟 임베딩을 정규화한 후 추론을 가해 도메인 정합성을 맞추었습니다.\n\n")
        
        for t_name in targets:
            if s_name == t_name:
                continue
                
            print(f"Running inference: Source = {s_name} -> Target = {t_name}")
            X_emb, y_valid, valid_coords, shape = data_cache[t_name]
            inference_results = run_cross_inference(s_name, t_name, X_emb, y_valid, valid_coords, shape)
            
            output_md.append(f"### 타겟 데이터셋: {t_name}\n")
            output_md.append("| 타겟 실제 클래스 (Actual Class) | 1순위 예측 클래스 (확률) | 2순위 예측 클래스 (확률) | 3순위 예측 클래스 (확률) |\n")
            output_md.append("| :--- | :--- | :--- | :--- |\n")
            
            for t_cls, res in inference_results.items():
                top3 = res["top3"]
                p1 = f"{top3[0]['class_name']} ({top3[0]['prob']*100:.2f}%)"
                p2 = f"{top3[1]['class_name']} ({top3[1]['prob']*100:.2f}%)" if len(top3) > 1 else "-"
                p3 = f"{top3[2]['class_name']} ({top3[2]['prob']*100:.2f}%)" if len(top3) > 2 else "-"
                output_md.append(f"| **{res['target_class_name']}** | {p1} | {p2} | {p3} |\n")
            
            output_md.append("\n")
            
            # Print specifically for Botswana target
            if t_name == "Botswana":
                print(f"\n--- {s_name} GNN Model -> Botswana Target Predictions ---")
                for t_cls, res in inference_results.items():
                    top1 = res["top3"][0]
                    print(f"Botswana Class: {res['target_class_name']:<25} | Predicted: {top1['class_name']:<35} ({top1['prob']*100:.2f}%)")
                print("-" * 50)
                
        output_md.append("---\n")
        
    # Write report
    report_path = "reports/gnn_cross_dataset_inference_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(output_md)
    print(f"\nSuccessfully generated inference analysis report at {report_path}")

if __name__ == "__main__":
    main()
