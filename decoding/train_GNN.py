import os
import numpy as np
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp

# --- Custom GNN Model Definition ---
class GraphConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, adj_matrix):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Add self-loops and Normalize: D^(-1) A
        A = adj_matrix + np.eye(adj_matrix.shape[0])
        D = np.sum(A, axis=1)
        D_inv = 1.0 / D
        D_inv[np.isinf(D_inv)] = 0.
        A_norm = (A.T * D_inv).T
        
        self.register_buffer('adj', torch.FloatTensor(A_norm))

    def forward(self, x):
        x_trans = self.linear(x) 
        out = torch.matmul(self.adj, x_trans) 
        return out

class LatentBroadcastGNN(nn.Module):
    def __init__(self, input_dim, num_voxels, adj_matrix, hidden_dim=256): # Default upped to 256
        super(LatentBroadcastGNN, self).__init__()
        
        self.num_voxels = num_voxels
        
        # 1. Global Stimulus Encoder (Capacity Increased)
        self.stim_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2), # Added LayerNorm for stability
            nn.GELU(), # ReLU -> GELU (Better for Transformers/Modern DL)
            nn.Dropout(0.3), # Increased Dropout for larger model
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 2. Voxel Embeddings
        self.voxel_embeddings = nn.Parameter(torch.randn(num_voxels, hidden_dim) * 0.01)
        
        # 3. GNN Layers
        self.gnn1 = GraphConvLayer(hidden_dim, hidden_dim, adj_matrix)
        self.act1 = nn.Tanh() 
        self.gnn2 = GraphConvLayer(hidden_dim, hidden_dim, adj_matrix)
        self.act2 = nn.Tanh()
        
        # 4. Readout
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, stim_features):
        # stim_features: (Batch, Input_Dim)
        
        global_state = self.stim_encoder(stim_features) # (Batch, Hidden)
        
        # Broadcast & Fuse
        global_state_expanded = global_state.unsqueeze(1) 
        voxel_emb_expanded = self.voxel_embeddings.unsqueeze(0)
        
        # Fusion
        x = global_state_expanded * voxel_emb_expanded 
        
        # Graph Convolution with Residuals
        x = self.gnn1(x)
        x = self.act1(x) + x 
        x = self.gnn2(x)
        x = self.act2(x) + x 
        
        # Readout
        pred_resp = self.readout(x).squeeze(-1)
        return pred_resp

# --- Main Training Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--gpt", type=str, default="perceived")
    parser.add_argument("--batch_size", type=int, default=64) # Increased for A6000
    parser.add_argument("--epochs", type=int, default=50)     # Increased Epochs
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256) # New Argument
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training GNN on {device} (A6000 Optimized Mode)")
    print(f"    - Batch Size: {args.batch_size}")
    print(f"    - Hidden Dim: {args.hidden_dim}")
    print(f"    - Epochs:     {args.epochs}")

    # 1. Load Adjacency Matrix
    adj_path = os.path.join(config.MODEL_DIR, args.subject, "adjacency_matrix.npz")
    adj_data = np.load(adj_path)
    adj_matrix = adj_data['adj']
    voxels = np.load(os.path.join(config.MODEL_DIR, args.subject, "encoding_model_perceived.npz"))["voxels"]
    
    # 2. Prepare Data
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
    
    print("[*] Extracting Stimulus Features & Responses...")
    rstim, tr_stats, word_stats = get_stim(stories, features)
    rresp = get_resp(args.subject, stories, stack=True)[:, voxels]
    
    # Normalize Response (Z-score)
    resp_mean = rresp.mean(axis=0)
    resp_std = rresp.std(axis=0)
    rresp = (rresp - resp_mean) / (resp_std + 1e-8)
    
    # Convert to Tensor
    dataset = TensorDataset(torch.FloatTensor(rstim), torch.FloatTensor(rresp))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 3. Initialize Model with Custom Hidden Dim
    model = LatentBroadcastGNN(
        input_dim=rstim.shape[1], 
        num_voxels=rresp.shape[1], 
        adj_matrix=adj_matrix,
        hidden_dim=args.hidden_dim # Applying argument
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # Switch to AdamW
    
    # 4. Training Loop
    print("[*] Starting Training...")
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
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f}")
        
    # 5. Save Model (Includes hidden_dim in metadata for decoder compatibility)
    save_path = os.path.join(config.MODEL_DIR, args.subject, f"gnn_model_{args.gpt}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tr_stats': tr_stats,
        'word_stats': word_stats,
        'resp_mean': resp_mean,
        'resp_std': resp_std,
        'voxels': voxels,
        'hidden_dim': args.hidden_dim # IMPORTANT: Save this for the decoder!
    }, save_path)
    print(f"[*] Model saved to {save_path}")