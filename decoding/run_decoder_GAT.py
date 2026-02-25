import os
import numpy as np
import json
import argparse
import h5py
import torch
import gc  # [Added] Garbage Collector
import config
from GPT import GPT
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel
from StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
from utils_stim import predict_word_rate, predict_word_times
from train_GAT import GATModel

# --- Custom GAT Encoding Model Wrapper ---
class GATEncodingModel:
    def __init__(self, model, resp, noise_model, resp_mean, resp_std, device="cuda"):
        self.model = model
        self.model.eval()
        self.resp = torch.from_numpy(resp).float().to(device)
        self.device = device
        self.sigma = noise_model
        
        # [Added] Denormalization Statistics
        self.resp_mean = torch.from_numpy(resp_mean).float().to(device)
        self.resp_std = torch.from_numpy(resp_std).float().to(device)
        
        self.precision = None
        self.inference_batch_size = 16 # [Added] OOM Prevention

    def set_shrinkage(self, alpha):
        """Compute precision matrix from baseline noise model"""
        sigma_shrunk = self.sigma * (1 - alpha) + np.eye(len(self.sigma)) * alpha
        # Matrix inversion safety
        try:
            precision = np.linalg.inv(sigma_shrunk)
        except np.linalg.LinAlgError:
            sigma_shrunk += np.eye(len(self.sigma)) * 1e-5
            precision = np.linalg.inv(sigma_shrunk)
        self.precision = torch.from_numpy(precision).float().to(self.device)

    def prs(self, stim, trs):
        """Compute P(R | S) using GAT predictions with Mini-batch & Denormalization"""
        n_variants, n_trs, n_features = stim.shape
        
        # Flatten input: (Total_Samples, Features)
        total_samples = n_variants * n_trs
        stim_flat = stim.reshape(total_samples, n_features)
        
        pred_resp_list = []
        
        with torch.no_grad():
            # [Added] Mini-batch Loop
            for i in range(0, total_samples, self.inference_batch_size):
                batch_stim = stim_flat[i : i + self.inference_batch_size].to(self.device)
                
                # 1. GAT Prediction (Normalized)
                batch_pred_norm = self.model(batch_stim)
                
                # 2. Denormalization (Restore to original fMRI scale)
                # Prediction = (Norm_Output * Std) + Mean
                batch_pred_raw = batch_pred_norm * self.resp_std + self.resp_mean
                
                pred_resp_list.append(batch_pred_raw)
            
            # Concatenate results
            pred_resp_flat = torch.cat(pred_resp_list, dim=0)
            
            # Reshape back
            pred_resp = pred_resp_flat.view(n_variants, n_trs, -1)
            
            # 3. Calculate Residuals
            diff = pred_resp - self.resp[trs].unsqueeze(0)
            
            # 4. Mahalanobis Distance
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
    word_rate_voxels = "auditory" 

    print(f"[*] Starting GAT Decoding for {args.subject} / {args.task}...")

    # 1. Load Responses (FULL)
    hf = h5py.File(os.path.join(config.DATA_TEST_DIR, "test_response", args.subject, args.experiment, args.task + ".hf5"), "r")
    full_resp = np.nan_to_num(hf["data"][:]) 
    hf.close()
    
    # 2. Load GPT & LM
    with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "model"), vocab=gpt_vocab, device=config.GPT_DEVICE)
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass=config.LM_MASS, nuc_ratio=config.LM_RATIO)

    # 3. Load Baseline Models
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle=True)
    
    baseline_em = np.load(os.path.join(load_location, "encoding_model_%s.npz" % gpt_checkpoint))
    noise_model = baseline_em["noise_model"]
    
    # 4. Load Trained GAT Model
    gat_path = os.path.join(load_location, f"gat_model_{gpt_checkpoint}.pt")
    print(f"[*] Loading GAT from {gat_path}")
    
    checkpoint = torch.load(gat_path, weights_only=False)
    
    tr_stats = checkpoint['tr_stats']
    word_stats = checkpoint['word_stats']
    voxels = checkpoint['voxels']
    
    # [Added] Load Mean/Std for Denormalization
    # 만약 train_GAT.py를 수정하기 전이라 이 값이 없다면 에러가 납니다.
    # 그럴 경우 임시방편으로 full_resp의 평균/표준편차를 사용해야 하지만, 
    # 학습 데이터 통계(checkpoint)를 쓰는 것이 원칙적으로 맞습니다.
    resp_mean = checkpoint.get('resp_mean')
    resp_std = checkpoint.get('resp_std')
    
    # Fallback if checkpoint doesn't have stats (Backward Compatibility)
    if resp_mean is None:
        print("[!] Warning: resp_mean not found in checkpoint. Using placeholder (Might degrade performance).")
        resp_mean = np.zeros(len(voxels))
        resp_std = np.ones(len(voxels))

    # Re-initialize GAT
    adj_path = os.path.join(config.MODEL_DIR, args.subject, "adjacency_matrix.npz")
    adj_matrix = np.load(adj_path)['adj']
    
    # Dynamic Input Dim
    input_dim = tr_stats[0].shape[0] * 4 # 3072
    num_voxels = len(voxels)
    
    # Initialize GATModel
    gat_model = GATModel(input_dim, num_voxels, adj_matrix, hidden_dim=64).to(config.EM_DEVICE)
    gat_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Prepare inputs
    gat_resp = full_resp[:, voxels]
    
    # Wrapper (Pass mean/std)
    em = GATEncodingModel(gat_model, gat_resp, noise_model, resp_mean, resp_std, device=config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    
    # 5. Decode
    print("[*] Predicting Word Rates...")
    word_rate = predict_word_rate(full_resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
    word_times, tr_times = predict_word_times(word_rate, full_resp, starttime=-10)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)

    # Beam Width 200 (Adjust if slow)
    decoder = Decoder(word_times, 200)
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device=config.SM_DEVICE)
    
    print("[*] Decoding... (This will take time)")
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
            
            # GAT Inference
            likelihoods = em.prs(stim, trs)
            
            local_extensions = [Hypothesis(parent=hyp, extension=x) for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose=False)
        
        # [Added] Memory Cleanup
        if sample_index % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            if decoder.beam and decoder.beam[0].words:
                print(f"Step {sample_index}/{len(word_times)}: ...{' '.join(decoder.beam[0].words[-5:])} [Mem Cleaned]")
            else:
                print(f"Step {sample_index}/{len(word_times)}: [Start] [Mem Cleaned]")

    # Save
    save_location = os.path.join(config.RESULT_DIR, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok=True)
    save_path = os.path.join(save_location, args.task + "_GAT") 
    decoder.save(save_path)
    print(f"[*] GAT Decoding results saved to {save_path}.npz")