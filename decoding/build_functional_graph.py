import os
import numpy as np
import json
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import config
from utils_resp import get_resp

def apply_data_driven_parcellation(resp_data, num_rois=400):
    """
    K-Means 클러스터링을 사용하여 10,000개의 복셀을 반응 패턴이 비슷한 400개의 ROI로 압축합니다.
    """
    print(f"[*] Clustering {resp_data.shape[1]} voxels into {num_rois} data-driven ROIs...")
    
    # 1. Z-score 정규화 (패턴 기반 클러스터링을 위해 필수)
    # resp_data: (Time, Voxels) -> scaler는 각 Voxel의 Time 시계열을 평균 0, 분산 1로 만듦
    scaler = StandardScaler()
    resp_normalized = scaler.fit_transform(resp_data)
    
    # 2. Voxel 간의 유사도를 보기 위해 Transpose -> (Voxels, Time)
    voxel_features = resp_normalized.T 
    
    # 3. K-Means 클러스터링 수행
    kmeans = KMeans(n_clusters=num_rois, random_state=42, n_init='auto')
    mapping = kmeans.fit_predict(voxel_features) # 각 복셀이 속한 ROI 인덱스 (0~399) 반환
    
    # 4. 각 ROI별 평균 시계열 계산 (원본 스케일 데이터 사용)
    roi_resp = np.zeros((resp_data.shape[0], num_rois))
    for roi_idx in range(num_rois):
        voxel_indices = np.where(mapping == roi_idx)[0]
        if len(voxel_indices) > 0:
            roi_resp[:, roi_idx] = np.mean(resp_data[:, voxel_indices], axis=1)
            
    return roi_resp, mapping

def build_functional_graph(roi_resp, threshold=0.3):
    """
    ROI 시계열 데이터 간의 피어슨 상관계수 행렬을 계산하여 인접 행렬(Adjacency Matrix)을 만듭니다.
    """
    print(f"[*] Calculating Pearson Correlation Matrix for {roi_resp.shape[1]} ROIs...")
    
    # 피어슨 상관계수 계산
    corr_matrix = np.corrcoef(roi_resp, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix)
    
    # Thresholding: 임계값 이상의 상관관계만 엣지(Edge)로 남김
    adj_matrix = np.where(corr_matrix > threshold, corr_matrix, 0.0)
    
    # Self-loop 제거 (GAT 내부에서 처리하거나 구조적 특징을 위해 0으로 둠)
    np.fill_diagonal(adj_matrix, 0.0)
    
    return adj_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.3, help="Correlation threshold for connectivity")
    parser.add_argument("--num_rois", type=int, default=400, help="Target number of ROIs")
    args = parser.parse_args()

    # 1. 학습용 Story 데이터 로드
    sessions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20]
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f) 
    for sess in sessions:
        stories.extend(sess_to_story[str(sess)])

    # 2. Voxel 인덱스 로드 (기존 10,000개 복셀)
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    baseline_em = np.load(os.path.join(load_location, "encoding_model_perceived.npz"))
    voxels = baseline_em["voxels"]

    print("[*] Loading full fMRI Responses for Functional Graph...")
    # (Total_Time, 10000) 형태의 데이터
    rresp = get_resp(args.subject, stories, stack=True)[:, voxels] 

    # 3. 복셀 압축 (10000 -> 400)
    roi_resp, mapping = apply_data_driven_parcellation(rresp, num_rois=args.num_rois)

    # 4. 400 노드 간의 기능적 연결망 추출
    adj_matrix = build_functional_graph(roi_resp, threshold=args.threshold)
    
    # 5. 매핑 정보와 인접 행렬 저장
    save_path = os.path.join(load_location, "functional_adjacency_400.npz")
    np.savez(save_path, adj=adj_matrix, mapping=mapping)
    print(f"[*] Functional Adjacency Matrix ({args.num_rois}x{args.num_rois}) & Mapping saved to {save_path}")
    
    # 밀집도(Sparsity) 리포트
    total_edges = args.num_rois * args.num_rois
    active_edges = np.count_nonzero(adj_matrix)
    print(f"[*] Graph Sparsity: {active_edges} / {total_edges} edges remain ({active_edges/total_edges*100:.2f}%).")