import os
import numpy as np
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp

# --- 정통 GCN Layer (Phase 4 R-GCN을 위한 완벽한 베이스라인) ---
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super(GraphConvLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # [GCN 핵심] Symmetric Normalization: D^{-1/2} * A * D^{-1/2}
        # 1. Self-loop 추가 (자기 자신의 정보도 유지하기 위함)
        A_tilde = adj_matrix + np.eye(adj_matrix.shape[0])
        
        # 2. Degree 행렬 계산
        D = np.sum(A_tilde, axis=1)
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0 # 0으로 나누는 것 방지
        D_inv_sqrt_mat = np.diag(D_inv_sqrt)
        
        # 3. 정규화된 인접 행렬
        A_norm = np.dot(D_inv_sqrt_mat, np.dot(A_tilde, D_inv_sqrt_mat))
        
        # GPU 메모리에 올리기 위해 buffer로 등록
        self.register_buffer('adj_norm', torch.FloatTensor(A_norm))

    def forward(self, h):
        # h: (Batch, 400, In_Features)
        Wh = self.W(h) # (Batch, 400, Out_Features)
        
        # Graph Convolution 연산: Normalized_A @ Wh
        # self.adj_norm: (400, 400)
        # Wh: (Batch, 400, Out) -> torch.matmul이 자동으로 브로드캐스팅 처리
        h_prime = torch.matmul(self.adj_norm, Wh) 
        
        return F.elu(h_prime)

class Phase3GCNModel(nn.Module):
    def __init__(self, input_dim, num_voxels, num_rois, adj_matrix, mapping, hidden_dim=64, dropout=0.3):
        super(Phase3GCNModel, self).__init__()
        
        # 1. Global Stimulus Encoder
        self.stim_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.roi_embeddings = nn.Parameter(torch.randn(num_rois, hidden_dim) * 0.01)
        
        # 2. [수정됨] GCN Layers (400 노드)
        self.gcn1 = GraphConvLayer(hidden_dim, hidden_dim, adj_matrix)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim, adj_matrix)
        self.skip1 = nn.Linear(hidden_dim, hidden_dim)
        self.skip2 = nn.Linear(hidden_dim, hidden_dim)

        # 3. Unpooling & Readout
        self.register_buffer('mapping', torch.LongTensor(mapping))
        self.voxel_specific_transform = nn.Parameter(torch.randn(num_voxels, hidden_dim) * 0.01)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, stim_features):
        global_state = self.stim_encoder(stim_features)
        x_roi = global_state.unsqueeze(1) + self.roi_embeddings.unsqueeze(0)
        
        # GCN Message Passing
        x_res = x_roi
        x_roi = self.gcn1(x_roi)
        x_roi = x_roi + self.skip1(x_res)
        
        x_res = x_roi
        x_roi = self.gcn2(x_roi)
        x_roi = x_roi + self.skip2(x_res)
        
        # Unpooling (400 -> 10,000)
        x_voxel = x_roi[:, self.mapping, :] 
        
        x_voxel = x_voxel + self.voxel_specific_transform.unsqueeze(0)
        pred_resp = self.readout(x_voxel).squeeze(-1) 
        
        return pred_resp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--gpt", type=str, default="perceived")
    parser.add_argument("--batch_size", type=int, default=32) # GCN은 가벼우니 32로!
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4) 
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Phase 3 func GCN on {device} (Batch: {args.batch_size})")

    # 1. Data Load
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    
    graph_data = np.load(os.path.join(load_location, "functional_adjacency_400.npz"))
    adj_matrix = graph_data['adj']
    mapping = graph_data['mapping']
    
    baseline_em = np.load(os.path.join(load_location, "encoding_model_perceived.npz"))
    voxels = baseline_em["voxels"]
    
    sessions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20]
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f) 
    for sess in sessions:
        stories.extend(sess_to_story[str(sess)])
        
    with open(os.path.join(config.DATA_LM_DIR, args.gpt, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, args.gpt, "model"), vocab=gpt_vocab, device=config.GPT_DEVICE)
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    
    print("[*] Extracting Data...")
    rstim, tr_stats, word_stats = get_stim(stories, features)
    rresp = get_resp(args.subject, stories, stack=True)[:, voxels]
    
    resp_mean = rresp.mean(axis=0)
    resp_std = rresp.std(axis=0)
    rresp = (rresp - resp_mean) / (resp_std + 1e-8)
    
    input_dim = tr_stats[0].shape[0] * len(config.STIM_DELAYS) # 3072
    num_voxels = len(voxels)
    num_rois = adj_matrix.shape[0] # 400
    hidden_dim = 64
    
    dataset = TensorDataset(torch.FloatTensor(rstim), torch.FloatTensor(rresp))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Model Init
    model = Phase3GCNModel(
        input_dim=input_dim, 
        num_voxels=num_voxels, 
        num_rois=num_rois,
        adj_matrix=adj_matrix, 
        mapping=mapping,
        hidden_dim=hidden_dim
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print("[*] Starting Phase 3 func GCN Training...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            output = model(bx)
            loss = criterion(output, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(dataloader):.6f}")
        
    # 3. Save Checkpoint
    save_path = os.path.join(config.MODEL_DIR, args.subject, f"gcn_phase3_func_{args.gpt}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tr_stats': tr_stats, 
        'word_stats': word_stats, 
        'voxels': voxels,
        'resp_mean': resp_mean,
        'resp_std': resp_std,
        'hidden_dim': hidden_dim,
        'num_rois': num_rois
    }, save_path)
    print(f"[*] Phase 3 GCN Model saved to {save_path}")