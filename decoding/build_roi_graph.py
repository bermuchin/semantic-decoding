import os
import json
import torch
import numpy as np
import config
import argparse

def build_roi_graph(subject):
    print(f"[*] Starting ROI-based Graph Construction for {subject}...")
    
    # 1. GAT 모델에서 사용한 1차원 복셀 인덱스 로드
    checkpoint_path = os.path.join(config.MODEL_DIR, subject, "gat_func_perceived.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[!] 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    selected_voxels = checkpoint["voxels"]
    num_voxels = len(selected_voxels)
    print(f"[*] Loaded {num_voxels} target voxels from GAT checkpoint.")
    
    # 2. 빠른 검색을 위한 매핑 딕셔너리 생성 (원본 복셀 번호 -> 0~1999 배열 인덱스)
    orig_to_idx = {int(orig_v): idx for idx, orig_v in enumerate(selected_voxels)}
    
    # 3. ROI JSON 파일 로드 (예: s3.json)
    roi_path = os.path.join(config.DATA_TRAIN_DIR, "ROIs", f"{subject.lower()}.json")
    if not os.path.exists(roi_path):
        # 혹시 파일명이 대문자 S3.json일 경우를 대비한 처리
        roi_path = os.path.join(config.DATA_TRAIN_DIR, "ROIs", f"{subject.upper()}.json")
        
    with open(roi_path, "r") as f:
        roi_data = json.load(f)
        
    print(f"[*] Loaded {len(roi_data)} ROIs from {roi_path}")
    
    # 4. 인접 행렬 초기화 (자기 자신과의 연결은 1.0으로 시작)
    adj_matrix = np.eye(num_voxels)
    
    # 5. 같은 ROI에 속한 복셀들끼리 엣지(Edge) 연결
    matched_voxels = set()
    for roi_name, roi_voxels in roi_data.items():
        # 현재 ROI에 속하면서, 우리가 선택한 2000개 복셀 안에도 있는 교집합 찾기
        valid_indices = [orig_to_idx[v] for v in roi_voxels if v in orig_to_idx]
        
        if len(valid_indices) > 0:
            print(f"  - [ROI: {roi_name}]: {len(valid_indices)} voxels matched.")
            matched_voxels.update(valid_indices)
            
            # 교집합에 속한 복셀들끼리 모두 완전 연결 (클러스터 형성)
            for i in valid_indices:
                for j in valid_indices:
                    adj_matrix[i, j] = 1.0
                    
    print(f"[*] Total connected voxels out of {num_voxels}: {len(matched_voxels)}")
    
    # 6. 저장
    save_dir = os.path.join(config.MODEL_DIR, subject)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "roi_adjacency_matrix.npz")
    
    np.savez(save_path, adj=adj_matrix)
    print(f"[*] ROI Graph successfully saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    args = parser.parse_args()
    
    build_roi_graph(args.subject)