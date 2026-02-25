import os
import numpy as np
import json
import argparse
import h5py
import torch
import gc
import config
from GPT import GPT
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel
from StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
from utils_stim import predict_word_rate, predict_word_times

# 학습 스크립트가 있는 경로에서 클래스 임포트 (파일이 같은 폴더에 있어야 함)
try:
    from train_GNN import LatentBroadcastGNN
except ImportError:
    raise ImportError("train_GNN.py not found. Please ensure it is in the same directory.")

# --- 수정된 GNNEncodingModel (Mini-batch 적용) ---
class GNNEncodingModel:
    def __init__(self, model, resp, noise_model, device="cuda"):
        self.model = model
        self.model.eval()
        self.resp = torch.from_numpy(resp).float().to(device)
        self.device = device
        self.sigma = noise_model
        self.precision = None
        # [설정] 추론 시 GNN에 태울 최대 배치 사이즈 (학습 때와 동일하게 512 권장)
        self.inference_batch_size = 1024

    def set_shrinkage(self, alpha):
        """Compute precision matrix from baseline noise model"""
        print(f"[*] Computing Precision Matrix (Shrinkage Alpha={alpha})...")
        sigma_shrunk = self.sigma * (1 - alpha) + np.eye(len(self.sigma)) * alpha
        
        try:
            precision = np.linalg.inv(sigma_shrunk)
        except np.linalg.LinAlgError:
            print("[!] Matrix inversion failed. Adding jitter.")
            sigma_shrunk += np.eye(len(self.sigma)) * 1e-5
            precision = np.linalg.inv(sigma_shrunk)
            
        self.precision = torch.from_numpy(precision).float().to(self.device)

    def prs(self, stim, trs):
        """Compute P(R | S) using GNN predictions with Mini-batching"""
        # stim shape: (n_variants, n_trs, n_features)
        n_variants, n_trs, n_features = stim.shape
        
        # 1. Flatten input: (Total_Samples, Features)
        total_samples = n_variants * n_trs
        stim_flat = stim.reshape(total_samples, n_features) # CPU or GPU tensor
        
        # 2. Mini-batch Processing (OOM 방지 핵심 로직)
        pred_resp_list = []
        
        with torch.no_grad():
            for i in range(0, total_samples, self.inference_batch_size):
                # 배치만큼 자르기
                batch_stim = stim_flat[i : i + self.inference_batch_size].to(self.device)
                
                # GNN Forward
                batch_pred = self.model(batch_stim) # (Batch, Voxels)
                
                # 결과 모으기 (GPU 메모리 절약을 위해 필요하다면 CPU로 이동, 여기선 GPU 유지)
                pred_resp_list.append(batch_pred)
            
            # 3. 전체 결과 합치기
            pred_resp_flat = torch.cat(pred_resp_list, dim=0) # (Total_Samples, Voxels)
            
            # 4. 원래 형태로 복원: (Variants, Time, Voxels)
            pred_resp = pred_resp_flat.view(n_variants, n_trs, -1)
            
            # 5. Residual & Likelihood 계산
            # self.resp[trs]: (Time, Voxels) -> Broadcast to (Variants, Time, Voxels)
            diff = pred_resp - self.resp[trs].unsqueeze(0)
            
            # Mahalanobis Distance
            term1 = torch.matmul(diff, self.precision)
            log_likelihoods = -0.5 * torch.sum(term1 * diff, dim=-1).sum(dim=-1)
            
            return log_likelihoods.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()
    
    gpt_checkpoint = "perceived"
    # Select word rate model based on experiment type (as per README/Config)
    if args.experiment in ["imagined_speech", "perceived_movies"]:
        word_rate_voxels = "speech"
    else:
        word_rate_voxels = "auditory"

    print(f"[*] Starting GNN Decoding for {args.subject} / {args.task}...")

    # 1. Load Responses
    resp_path = os.path.join(config.DATA_TEST_DIR, "test_response", args.subject, args.experiment, args.task + ".hf5")
    with h5py.File(resp_path, "r") as hf:
        full_resp = np.nan_to_num(hf["data"][:])
    
    # 2. Load GPT & LM
    with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)
        
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "model"), vocab=gpt_vocab, device=config.GPT_DEVICE)
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass=config.LM_MASS, nuc_ratio=config.LM_RATIO)

    # 3. Load Baseline Models (for Noise Model & Word Rate)
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle=True)
    
    # Load Baseline Encoding Model to get Noise Model
    baseline_em = np.load(os.path.join(load_location, "encoding_model_%s.npz" % gpt_checkpoint))
    noise_model = baseline_em["noise_model"]
    baseline_voxels = baseline_em["voxels"]
    
    # 4. Load Trained GNN Model
    gnn_path = os.path.join(load_location, f"gnn_model_{gpt_checkpoint}.pt")
    print(f"[*] Loading GNN from {gnn_path}")
    
    checkpoint = torch.load(gnn_path, weights_only=False)
    
    # Extract stats
    tr_stats = checkpoint["tr_stats"] # (mean, std)
    word_stats = checkpoint["word_stats"]
    gnn_voxels = checkpoint["voxels"]
    
    # [CRITICAL CHECK] Ensure Voxels Match
    if not np.array_equal(baseline_voxels, gnn_voxels):
        print("[!] WARNING: Baseline voxels and GNN voxels do not match perfectly.")
        print("    Adjusting noise model to match GNN voxels is complex.")
        print("    If specific indices differ, the noise model will be invalid.")
        # Strict Check (Recommended)
        # raise ValueError("Voxel mismatch between Baseline Noise Model and GNN Model")
    
    # Re-initialize GNN
    adj_path = os.path.join(config.MODEL_DIR, args.subject, "adjacency_matrix.npz")
    adj_matrix = np.load(adj_path)["adj"]
    # The code snippet `# Dynamic Input Dimension` and `# Dynamic Input Dimension (Corrected)` is
    # calculating the input dimension for the GNN model based on the statistics of the training data.
    

    # Dynamic Input Dimension (Corrected)
    n_features = tr_stats[0].shape[0] # 768
    n_delays = len(config.STIM_DELAYS) # 4 ([1, 2, 3, 4])
    input_dim = n_features * n_delays # 768 * 4 = 3072
    num_voxels = len(gnn_voxels)
    hidden_dim = checkpoint.get("hidden_dim", 512) # Use saved dim or default to 512
    
    gnn_model = LatentBroadcastGNN(
        input_dim,
        num_voxels,
        adj_matrix,
        hidden_dim=hidden_dim, 
    ).to(config.EM_DEVICE)
    gnn_model.load_state_dict(checkpoint["model_state_dict"])
    
    # Subset responses to match GNN voxels
    gnn_resp = full_resp[:, gnn_voxels]
    
    # Construct GNN Encoding Model Wrapper
    em = GNNEncodingModel(gnn_model, gnn_resp, noise_model, device=config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    
    # 5. Decode
    print("[*] Predicting Word Rates...")
    word_rate = predict_word_rate(full_resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
    
    # Standardize starttime based on experiment (Copied from original run_decoder.py)
    starttime = -10 if args.experiment == "perceived_speech" else 0
    word_times, tr_times = predict_word_times(word_rate, full_resp, starttime=starttime)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)

    print("[*] Setting Beam Width to 100 (High VRAM Usage Mode)")
    decoder = Decoder(word_times, 100) 
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device=config.SM_DEVICE)
    
    print("[*] Decoding... (This will take time)")
    for sample_index in range(len(word_times)):
        trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
        ncontext = decoder.time_window(sample_index, config.LM_TIME, floor=5)
        
        # Propose extensions
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        
        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if len(nuc) < 1: continue
            
            extend_words = [hyp.words + [x] for x in nuc]
            extend_embs = list(features.extend(extend_words))
            
            # Create Stimulus Features
            stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
            
            # GNN Inference (Likelihood Calculation)
            likelihoods = em.prs(stim, trs)
            
            local_extensions = [Hypothesis(parent=hyp, extension=x) for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        
        decoder.extend(verbose=False)
        
        # Memory Cleanup
        if sample_index % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            # Safety check: Ensure beam isn't empty
            if decoder.beam and decoder.beam[0].words:
                last_words = decoder.beam[0].words[-3:]
                print(f"Step {sample_index}/{len(word_times)}: ...{' '.join(last_words)} [Mem Cleaned]")
            else:
                print(f"Step {sample_index}/{len(word_times)}: [Start] [Mem Cleaned]")

    # Save Results
    save_location = os.path.join(config.RESULT_DIR, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok=True)
    save_path = os.path.join(save_location, args.task + "_GNN")
    decoder.save(save_path)
    print(f"[*] GNN Decoding results saved to {save_path}.npz")