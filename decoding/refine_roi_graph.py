import os
import numpy as np
import argparse
import config

def refine_roi_graph(subject, top_percent=20):
    print(f"[*] Starting ROI Graph Refinement (Edge Masking) for {subject}...")
    
    # 1. 두 개의 행렬 불러오기
    model_dir = os.path.join(config.MODEL_DIR, subject)
    roi_path = os.path.join(model_dir, "roi_adjacency_matrix.npz")
    func_path = os.path.join(model_dir, "adjacency_matrix.npz") # Phase 3에서 쓰던 기존 기능적 그래프
    
    if not os.path.exists(roi_path):
        raise FileNotFoundError(f"[!] ROI 행렬을 찾을 수 없습니다: {roi_path}")
    if not os.path.exists(func_path):
        raise FileNotFoundError(f"[!] 기능적 연결성 행렬을 찾을 수 없습니다: {func_path}")
        
    roi_adj = np.load(roi_path)["adj"]
    func_adj = np.load(func_path)["adj"]
    
    print(f"[*] Loaded ROI Adjacency Matrix: {roi_adj.shape}")
    print(f"[*] Loaded Functional Adjacency Matrix: {func_adj.shape}")
    
    # 2. 아다마르 곱 (Hadamard Product) - 마스킹
    # 같은 ROI에 속한(roi_adj=1) 위치의 기능적 상관계수(func_adj)만 남기고 나머지는 0이 됩니다.
    masked_adj = roi_adj * func_adj
    
    # 3. 임계값(Thresholding) 적용: ROI 내부에서도 진짜 친한 상위 K%만 남기기
    # 대각 성분(자기 자신과의 연결)과 0인 부분은 제외하고 임계값을 계산합니다.
    # np.triu를 써서 중복되는 절반을 날리고 계산하면 더 정확합니다.
    upper_triangle = np.triu(masked_adj, k=1)
    non_zero_edges = upper_triangle[upper_triangle > 0]
    
    if len(non_zero_edges) == 0:
        print("[!] 경고: ROI 내부에 기능적 연결성이 0보다 큰 엣지가 하나도 없습니다.")
        final_adj = masked_adj
    else:
        # 하위 (100 - top_percent)% 의 값을 찾아 커트라인으로 설정
        threshold = np.percentile(non_zero_edges, 100 - top_percent)
        print(f"[*] Cutting off edges below correlation threshold: {threshold:.4f} (Top {top_percent}%)")
        
        # 커트라인을 넘는 엣지만 남기고 나머지는 0으로 만듦
        final_adj = np.where(masked_adj >= threshold, masked_adj, 0.0)
    
    # 4. 자기 자신과의 연결(Self-loop)은 무조건 1.0으로 보장
    np.fill_diagonal(final_adj, 1.0)
    
    # 통계 출력
    total_possible_edges = (2000 * 1999) / 2
    survived_edges = np.count_nonzero(np.triu(final_adj, k=1))
    print(f"[*] Refinement Complete!")
    print(f"  - Total possible connections: {total_possible_edges:,}")
    print(f"  - Survived core connections in ROIs: {survived_edges:,}")
    
    # 5. 정제된 행렬 저장
    save_path = os.path.join(model_dir, "refined_roi_adjacency_matrix.npz")
    np.savez(save_path, adj=final_adj)
    print(f"[*] Refined ROI Graph successfully saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--top", type=int, default=20, help="남길 상위 엣지의 퍼센티지 (기본 20%)")
    args = parser.parse_args()
    
    refine_roi_graph(args.subject, top_percent=args.top)