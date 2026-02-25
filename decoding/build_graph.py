import os
import numpy as np
import json
import argparse
import h5py
from tqdm import tqdm

import config
from utils_resp import get_resp

def compute_adjacency(subject, threshold=0.5):
    print(f"[*] Building Graph for Subject: {subject} with Threshold: {threshold}")
    
    # 1. Load Training Data (Sessions defined in config or default)
    # We use the same sessions as in train_EM.py
    sessions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20]
    
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    for sess in sessions:
        stories.extend(sess_to_story[str(sess)])
    
    # 2. Load Top Voxels (Using the indices from the trained Encoding Model)
    # We must build the graph ONLY for the voxels used in the model
    save_location = os.path.join(config.MODEL_DIR, subject)
    # Load perceived encoding model to get the voxel indices
    em_path = os.path.join(save_location, "encoding_model_perceived.npz")
    
    if not os.path.exists(em_path):
        raise FileNotFoundError(f"Encoding model not found at {em_path}. Please run train_EM.py first.")
        
    print(f"[*] Loading voxel indices from {em_path}...")
    encoding_model = np.load(em_path)
    vox_indices = encoding_model["voxels"] # shape: (10000,)
    
    # 3. Load Response Data for these voxels
    print("[*] Loading fMRI responses...")
    # stack=True returns (Time, All_Voxels), we select specific voxels
    resp = get_resp(subject, stories, stack=True)[:, vox_indices] 
    print(f"[*] Response shape: {resp.shape}")
    
    # 4. Compute Correlation Matrix (Functional Connectivity)
    print("[*] Computing Correlation Matrix (this may take a while)...")
    # Row-var=False because rows are time, cols are variables(voxels)
    # We want corr between columns
    corr_matrix = np.corrcoef(resp, rowvar=False)
    print(f"[*] Correlation Matrix shape: {corr_matrix.shape}")
    
    # 5. Thresholding to create Adjacency Matrix
    # Remove self-loops (diagonal = 0)
    np.fill_diagonal(corr_matrix, 0)
    
    # Binary Adjacency (Unweighted for now, or Weighted if needed)
    adj_matrix = np.where(np.abs(corr_matrix) > threshold, 1, 0)
    
    # Check sparsity
    num_edges = np.sum(adj_matrix)
    density = num_edges / (adj_matrix.shape[0] * adj_matrix.shape[1])
    print(f"[*] Graph Density: {density*100:.2f}% (Threshold: {threshold})")
    print(f"[*] Total Edges: {num_edges}")

    # 6. Save
    save_path = os.path.join(save_location, "adjacency_matrix.npz")
    np.savez(save_path, adj=adj_matrix, corr=corr_matrix, threshold=threshold)
    print(f"[*] Saved Adjacency Matrix to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    compute_adjacency(args.subject, args.threshold)