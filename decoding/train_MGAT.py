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

# --- Multi-head GAT Layer (기존과 동일, 건드리지 않음) ---
class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix, n_heads=4, dropout=0.3, alpha=0.2, concat=True):
        super(MultiHeadGATLayer, self).__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.head_dim = out_features // n_heads
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        A_tilde = adj_matrix + np.eye(adj_matrix.shape[0])
        mask = torch.FloatTensor(A_tilde)
        
        mask = torch.where(mask > 0, torch.zeros_like(mask), torch.full_like(mask, float('-inf')))
        self.register_buffer('adj_mask', mask)

    def forward(self, h):
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.n_heads, self.head_dim)
        Wh = Wh.transpose(1, 2) 
        scores = torch.matmul(Wh, Wh.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = scores + self.adj_mask.unsqueeze(0).unsqueeze(0)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            h_prime = h_prime.transpose(1, 2).contiguous().view(B, N, self.out_features)
            return F.elu(h_prime)
        else:
            h_prime = h_prime.mean(dim=1)
            return h_prime

# --- 🌟 대망의 Multiplex-GAT 모델 🌟 ---
class MGATModel(nn.Module):
    # [수정됨] 행렬을 두 개(adj_func, adj_roi) 입력받습니다.
    def __init__(self, input_dim, num_voxels, adj_func, adj_roi, hidden_dim=128, n_heads=4, dropout=0.3):
        super(MGATModel, self).__init__()
        
        self.stim_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.voxel_embeddings = nn.Parameter(torch.randn(num_voxels, hidden_dim) * 0.01)
        
        # [수정됨] 평행 우주 (Parallel Pathways) 생성
        # Track A: 기능적 그래프 전담
        self.func_gat1 = MultiHeadGATLayer(hidden_dim, hidden_dim, adj_func, n_heads=n_heads, dropout=dropout)
        self.func_gat2 = MultiHeadGATLayer(hidden_dim, hidden_dim, adj_func, n_heads=n_heads, dropout=dropout)
        
        # Track B: ROI 생물학적 사전지식 전담
        self.roi_gat1 = MultiHeadGATLayer(hidden_dim, hidden_dim, adj_roi, n_heads=n_heads, dropout=dropout)
        self.roi_gat2 = MultiHeadGATLayer(hidden_dim, hidden_dim, adj_roi, n_heads=n_heads, dropout=dropout)
        
        # [수정됨] 융합 레이어 (Fusion Layer)
        # 두 트랙에서 나온 결과(hidden_dim + hidden_dim)를 합쳐서 다시 원래 크기(hidden_dim)로 압축
        self.fusion1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion2 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.skip1 = nn.Linear(hidden_dim, hidden_dim)
        self.skip2 = nn.Linear(hidden_dim, hidden_dim)

        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, stim_features):
        global_state = self.stim_encoder(stim_features)
        x = global_state.unsqueeze(1) + self.voxel_embeddings.unsqueeze(0)
        
        # --- Layer 1 ---
        x_res1 = x
        h_func1 = self.func_gat1(x)    # Track A
        h_roi1 = self.roi_gat1(x)      # Track B
        
        # 두 세계관의 융합! (Concatenate -> Linear)
        h_fused1 = torch.cat([h_func1, h_roi1], dim=-1)
        x = self.fusion1(h_fused1)
        x = x + self.skip1(x_res1)     # 잔차 연결(ResNet) 유지
        
        # --- Layer 2 ---
        x_res2 = x
        h_func2 = self.func_gat2(x)
        h_roi2 = self.roi_gat2(x)
        
        h_fused2 = torch.cat([h_func2, h_roi2], dim=-1)
        x = self.fusion2(h_fused2)
        x = x + self.skip2(x_res2)
        
        # --- Readout ---
        pred_resp = self.readout(x).squeeze(-1) 
        return pred_resp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--gpt", type=str, default="perceived")
    
    # [VRAM 방어] M-GAT는 메모리를 2배로 쓰므로 batch_size를 더 줄이거나 유지합니다.
    parser.add_argument("--batch_size", type=int, default=2) 
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5) 
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Multiplex-GAT (M-GAT) on {device} (Batch: {args.batch_size})")

    # 1. Data Load
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    
    # [수정됨] 두 장의 지도를 모두 불러옵니다.
    print("[*] Loading Dual Graphs (Functional & ROI)...")
    adj_func = np.load(os.path.join(load_location, "adjacency_matrix.npz"))['adj']
    adj_roi = np.load(os.path.join(load_location, "refined_roi_adjacency_matrix.npz"))['adj']
    
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
    hidden_dim = 128
    
    dataset = TensorDataset(torch.FloatTensor(rstim), torch.FloatTensor(rresp))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Model Init
    # [수정됨] MGATModel 호출 및 행렬 두 개 전달
    model = MGATModel(
        input_dim=input_dim, 
        num_voxels=num_voxels,
        adj_func=adj_func, 
        adj_roi=adj_roi, 
        hidden_dim=hidden_dim,
        n_heads=4 
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print("[*] Starting Multiplex-GAT Training...")
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

        torch.cuda.empty_cache()
        
    # 3. Save Checkpoint
    save_path = os.path.join(config.MODEL_DIR, args.subject, f"mgat_func_{args.gpt}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tr_stats': tr_stats, 
        'word_stats': word_stats, 
        'voxels': voxels,
        'resp_mean': resp_mean,
        'resp_std': resp_std,
        'hidden_dim': hidden_dim,
        'n_heads': 4
    }, save_path)
    print(f"[*] M-GAT Model saved to {save_path}")