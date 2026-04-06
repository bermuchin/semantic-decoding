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

# --- Multi-head GAT Layer (Sparse Masking 적용) ---
class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix, n_heads=4, dropout=0.3, alpha=0.2, concat=True):
        super(MultiHeadGATLayer, self).__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.head_dim = out_features // n_heads # 각 헤드가 담당할 차원 (예: 256 // 4 = 64)
        self.concat = concat

        # 모든 헤드의 선형 변환을 한 번의 연산으로 처리 (효율성 극대화)
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        # [Sparse Attention 핵심] 연결되지 않은 엣지를 -inf로 마스킹
        A_tilde = adj_matrix + np.eye(adj_matrix.shape[0]) # 자기 자신과의 통신(Self-loop) 보장
        mask = torch.FloatTensor(A_tilde)
        
        # 인접 행렬 값이 0보다 크면 0을, 연결이 없으면 -inf를 할당
        # (Softmax에 -inf가 들어가면 어텐션 가중치가 완벽히 0이 되어 연산 배제 효과)
        mask = torch.where(mask > 0, torch.zeros_like(mask), torch.full_like(mask, float('-inf')))
        self.register_buffer('adj_mask', mask)

    def forward(self, h):
        B, N, _ = h.shape
        
        # 1. 선형 변환 및 헤드 분할 (Batch, Nodes, Heads, Head_dim)
        Wh = self.W(h).view(B, N, self.n_heads, self.head_dim)
        
        # 연산을 위해 (Batch, Heads, Nodes, Head_dim) 형태로 전치
        Wh = Wh.transpose(1, 2) 
        
        # 2. 어텐션 스코어 계산 (Dot-Product 방식) -> (Batch, Heads, Nodes, Nodes)
        scores = torch.matmul(Wh, Wh.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # 3. 마스킹 적용 (Broadcasting: adj_mask가 모든 배치와 헤드에 동일하게 적용됨)
        scores = scores + self.adj_mask.unsqueeze(0).unsqueeze(0)
        
        # 4. Softmax 및 정보 집계
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        h_prime = torch.matmul(attention, Wh) # (Batch, Heads, Nodes, Head_dim)
        
        if self.concat:
            # 여러 전문가(Heads)의 의견을 하나로 이어 붙임 (Concatenation)
            h_prime = h_prime.transpose(1, 2).contiguous().view(B, N, self.out_features)
            return F.elu(h_prime)
        else:
            # 마지막 레이어일 경우 의견을 평균 냄
            h_prime = h_prime.mean(dim=1)
            return h_prime

# --- 2000 Node GAT 모델 ---
class GATModel(nn.Module):
    def __init__(self, input_dim, num_voxels, adj_matrix, hidden_dim=256, n_heads=4, dropout=0.3):
        super(GATModel, self).__init__()
        
        self.stim_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.voxel_embeddings = nn.Parameter(torch.randn(num_voxels, hidden_dim) * 0.01)
        
        # 2층의 Multi-head GAT
        self.gat1 = MultiHeadGATLayer(hidden_dim, hidden_dim, adj_matrix, n_heads=n_heads, dropout=dropout, concat=True)
        self.gat2 = MultiHeadGATLayer(hidden_dim, hidden_dim, adj_matrix, n_heads=n_heads, dropout=dropout, concat=True)
        
        self.skip1 = nn.Linear(hidden_dim, hidden_dim)
        self.skip2 = nn.Linear(hidden_dim, hidden_dim)

        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, stim_features):
        global_state = self.stim_encoder(stim_features)
        x = global_state.unsqueeze(1) + self.voxel_embeddings.unsqueeze(0)
        
        x_res = x
        x = self.gat1(x)
        x = x + self.skip1(x_res)
        
        x_res = x
        x = self.gat2(x)
        x = x + self.skip2(x_res)
        
        pred_resp = self.readout(x).squeeze(-1) 
        return pred_resp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--gpt", type=str, default="perceived")
    
    # [주의] GAT는 메모리를 많이 쓰므로 기본 배치 사이즈를 8로 대폭 낮췄습니다.
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5) 
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training func GAT on {device} (Batch: {args.batch_size})")

    # 1. Data Load
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    
    graph_data = np.load(os.path.join(load_location, "adjacency_matrix.npz"))
    adj_matrix = graph_data['adj']
    
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
    hidden_dim = 256
    
    dataset = TensorDataset(torch.FloatTensor(rstim), torch.FloatTensor(rresp))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Model Init
    model = GATModel(
        input_dim=input_dim, 
        num_voxels=num_voxels,
        adj_matrix=adj_matrix, 
        hidden_dim=hidden_dim,
        n_heads=4 # 헤드 4개 지정
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print("[*] Starting func GAT Training...")
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
    save_path = os.path.join(config.MODEL_DIR, args.subject, f"gat_func_{args.gpt}.pt")
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
    print(f"[*] func GAT Model saved to {save_path}")