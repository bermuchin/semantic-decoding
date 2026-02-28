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

# 학습 코드에서 사용한 GCN 모델 클래스를 가져오기 위해 import
try:
    from train_GCN_func import Phase3GCNModel, GraphConvLayer
except ImportError:
    raise ImportError("train_GCN_func.py 파일이 같은 폴더에 있어야 합니다.")

# --- Custom GCN Encoding Model Wrapper ---
class GCNEncodingModel:
    def __init__(self, model, resp, noise_model, resp_mean, resp_std, device="cuda"):
        self.model = model
        self.model.eval()
        self.resp = torch.from_numpy(resp).float().to(device)
        self.device = device
        self.sigma = noise_model
        
        # 정규화 복원용 통계치
        self.resp_mean = torch.from_numpy(resp_mean).float().to(device)
        self.resp_std = torch.from_numpy(resp_std).float().to(device)
        
        self.precision = None
        self.inference_batch_size = 4 # 추론 시 빔서치 폭발 방지 (2080 Ti 최적화)

    def set_shrinkage(self, alpha):
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
        """Compute P(R | S) using GCN predictions with Denormalization"""
        n_variants, n_trs, n_features = stim.shape
        total_samples = n_variants * n_trs
        stim_flat = stim.reshape(total_samples, n_features)
        
        pred_resp_list = []
        
        with torch.no_grad():
            for i in range(0, total_samples, self.inference_batch_size):
                batch_stim = stim_flat[i : i + self.inference_batch_size].to(self.device)
                
                # 1. GCN Forward (정규화된 출력)
                batch_pred_norm = self.model(batch_stim)
                
                # 2. Denormalization (Z-Score 복원)
                batch_pred_raw = batch_pred_norm * self.resp_std + self.resp_mean
                pred_resp_list.append(batch_pred_raw)
            
            # Concatenate
            pred_resp_flat = torch.cat(pred_resp_list, dim=0)
            pred_resp = pred_resp_flat.view(n_variants, n_trs, -1)
            
            # 3. Residual & Likelihood
            diff = pred_resp - self.resp[trs].unsqueeze(0)
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
    if args.experiment in ["imagined_speech", "perceived_movies"]:
        word_rate_voxels = "speech"
    else:
        word_rate_voxels = "auditory"

    print(f"[*] Starting Phase 3 func GCN Decoding for {args.subject} / {args.task}...")

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

    # 3. Load Baseline Models (for noise_model and word_rate)
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle=True)
    
    baseline_em = np.load(os.path.join(load_location, "encoding_model_%s.npz" % gpt_checkpoint))
    noise_model = baseline_em["noise_model"]
    
    # 4. Load Trained Phase 3 GCN Model
    gcn_path = os.path.join(load_location, f"gcn_phase3_func_{gpt_checkpoint}.pt")
    print(f"[*] Loading Phase 3 func GCN from {gcn_path}")
    checkpoint = torch.load(gcn_path, weights_only=False)
    
    tr_stats = checkpoint["tr_stats"]
    word_stats = checkpoint["word_stats"]
    voxels = checkpoint["voxels"]
    resp_mean = checkpoint["resp_mean"]
    resp_std = checkpoint["resp_std"]
    hidden_dim = checkpoint["hidden_dim"]
    num_rois = checkpoint["num_rois"]
    
    # Load Functional Graph 
    graph_data = np.load(os.path.join(load_location, "functional_adjacency_400.npz"))
    adj_matrix = graph_data['adj']
    mapping = graph_data['mapping']
    
    input_dim = tr_stats[0].shape[0] * len(config.STIM_DELAYS)
    num_voxels = len(voxels)
    
    # Initialize GCN
    gcn_model = Phase3GCNModel(
        input_dim, num_voxels, num_rois, adj_matrix, mapping, hidden_dim=hidden_dim
    ).to(config.EM_DEVICE)
    gcn_model.load_state_dict(checkpoint["model_state_dict"])
    
    # Subset responses
    gcn_resp = full_resp[:, voxels]
    
    # Wrapper
    em = GCNEncodingModel(gcn_model, gcn_resp, noise_model, resp_mean, resp_std, device=config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    
    # 5. Decode
    print("[*] Predicting Word Rates...")
    word_rate = predict_word_rate(full_resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
    starttime = -10 if args.experiment == "perceived_speech" else 0
    word_times, tr_times = predict_word_times(word_rate, full_resp, starttime=starttime)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)

    decoder = Decoder(word_times, 200) # Beam Width 200
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device=config.SM_DEVICE)
    
    print("[*] Decoding... (Phase 3 func GCN Inference)")
    for sample_index in range(len(word_times)):
        trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
        ncontext = decoder.time_window(sample_index, config.LM_TIME, floor=5)
        
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        
        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if len(nuc) < 1: continue
            
            extend_words = [hyp.words + [x] for x in nuc]
            extend_embs = list(features.extend(extend_words))
            stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
            
            # GCN Inference
            likelihoods = em.prs(stim, trs)
            
            local_extensions = [Hypothesis(parent=hyp, extension=x) for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        
        decoder.extend(verbose=False)
        
        # Memory Cleanup
        if sample_index % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            if decoder.beam and decoder.beam[0].words:
                print(f"Step {sample_index}/{len(word_times)}: ...{' '.join(decoder.beam[0].words[-5:])} [Mem Cleaned]")
            else:
                print(f"Step {sample_index}/{len(word_times)}: [Start] [Mem Cleaned]")

    # Save Results
    save_location = os.path.join(config.RESULT_DIR, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok=True)
    save_path = os.path.join(save_location, args.task + "_GCN_func")
    decoder.save(save_path)
    print(f"[*] Phase 3 func GCN Decoding results saved to {save_path}.npz")