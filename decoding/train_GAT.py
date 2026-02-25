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

# --- GAT Layer Definition ---
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # [메모리 최적화 팁] Native GAT 구현에서 a 벡터는 사실상 concat attention에 쓰이는데,
        # 여기서는 Dot-product attention을 쓰고 있으므로 self.a는 사용되지 않을 수 있습니다.
        # 코드 일관성을 위해 유지하되, 실제 연산은 Dot-product로 진행합니다.
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        # Adjacency Mask
        self.register_buffer('adj', torch.FloatTensor(adj_matrix))

    def forward(self, h):
        # h: (Batch, Nodes, In_Features)
        Wh = torch.matmul(h, self.W) # (Batch, Nodes, Out_Features)
        
        # [메모리 주의 구간] Dense Attention Calculation
        # (Batch, N, Out) @ (Batch, Out, N) -> (Batch, N, N)
        # 배치 사이즈가 크면 여기서 OOM 발생
        scores = torch.matmul(Wh, Wh.transpose(1, 2)) / np.sqrt(self.out_features)
        
        # Masking: Adjacency가 0인 부분은 -1e9로 보내서 Softmax에서 0이 되게 함
        # -9e15는 float32에서 너무 작은 수라 NaN 위험이 있으므로 -1e9 정도로 안전하게 수정
        zero_vec = -1e9 * torch.ones_like(scores)
        attention = torch.where(self.adj > 0, scores, zero_vec)
        
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GATModel(nn.Module):
    def __init__(self, input_dim, num_voxels, adj_matrix, hidden_dim=64, dropout=0.3, alpha=0.2):
        super(GATModel, self).__init__()
        
        # Stimulus Encoder (3072 -> Hidden)
        self.stim_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.voxel_embeddings = nn.Parameter(torch.randn(num_voxels, hidden_dim) * 0.01)
        
        # GAT Layers
        self.gat1 = GraphAttentionLayer(hidden_dim, hidden_dim, adj_matrix, dropout=dropout, alpha=alpha, concat=True)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, adj_matrix, dropout=dropout, alpha=alpha, concat=False)
        
        # Skip Connections
        self.skip1 = nn.Linear(hidden_dim, hidden_dim)
        self.skip2 = nn.Linear(hidden_dim, hidden_dim)

        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, stim_features):
        global_state = self.stim_encoder(stim_features)
        
        global_state_expanded = global_state.unsqueeze(1) # (B, 1, H)
        voxel_emb_expanded = self.voxel_embeddings.unsqueeze(0) # (1, V, H)
        
        # Feature Fusion (Addition)
        x = global_state_expanded + voxel_emb_expanded 
        
        # GAT Layer 1
        x_res = x
        x = self.gat1(x)
        x = x + self.skip1(x_res) 
        
        # GAT Layer 2
        x_res = x
        x = self.gat2(x)
        x = x + self.skip2(x_res)
        
        pred_resp = self.readout(x).squeeze(-1)
        return pred_resp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--gpt", type=str, default="perceived")
    # [수정 1] 배치 사이즈 16
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--epochs", type=int, default=50) # Epoch 늘림
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training GAT on {device} (Batch: {args.batch_size})")

    # Load Data
    adj_path = os.path.join(config.MODEL_DIR, args.subject, "adjacency_matrix.npz")
    adj_matrix = np.load(adj_path)['adj']
    voxels = np.load(os.path.join(config.MODEL_DIR, args.subject, "encoding_model_perceived.npz"))["voxels"]
    
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
    
    # [수정 2] Input Dimension 명시적 계산 (안전장치)
    input_dim = tr_stats[0].shape[0] * len(config.STIM_DELAYS) # 768 * 4 = 3072
    print(f"[*] Input Dimension: {input_dim}")
    
    dataset = TensorDataset(torch.FloatTensor(rstim), torch.FloatTensor(rresp))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = GATModel(
        input_dim=input_dim, 
        num_voxels=rresp.shape[1], 
        adj_matrix=adj_matrix, 
        hidden_dim=64 # 메모리 부족 시 32로 줄여야 할 수도 있음
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) 
    
    print("[*] Starting Training (GAT)...")
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
        
    save_path = os.path.join(config.MODEL_DIR, args.subject, f"gat_model_{args.gpt}.pt")
    
    # [수정 3] 필수 정보(resp_mean, std, hidden_dim) 추가 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'tr_stats': tr_stats, 
        'word_stats': word_stats, 
        'voxels': voxels,
        'resp_mean': resp_mean, # 디코더에서 정규화 복원용
        'resp_std': resp_std,   # 디코더에서 정규화 복원용
        'hidden_dim': 64        # 모델 초기화용
    }, save_path)
    print(f"[*] GAT Model saved to {save_path}")