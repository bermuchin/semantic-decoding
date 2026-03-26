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

# [수정됨] 새로 만든 M-GAT 모델 클래스를 불러옵니다.
try:
    from train_MGAT import MGATModel
except ImportError:
    raise ImportError("train_MGAT.py 파일이 같은 폴더에 있어야 합니다.")

# --- MGAT Encoding Model Wrapper ---
class MGATEncodingModel:
    def __init__(self, model, resp, noise_model, resp_mean, resp_std, device="cuda"):
        self.model = model
        self.model.eval()
        self.resp = torch.from_numpy(resp).float().to(device)
        self.device = device
        self.sigma = noise_model
        
        self.resp_mean = torch.from_numpy(resp_mean).float().to(device)
        self.resp_std = torch.from_numpy(resp_std).float().to(device)
        
        self.precision = None
        self.inference_batch_size = 8

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
        n_variants, n_trs, n_features = stim.shape
        total_samples = n_variants * n_trs
        stim_flat = stim.reshape(total_samples, n_features)
        
        pred_resp_flat = torch.empty((total_samples, self.resp.shape[1]), device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            for i in range(0, total_samples, self.inference_batch_size):
                end_idx = min(i + self.inference_batch_size, total_samples)
                
                # 🌟 [수정 1] 슬라이싱 직후 반드시 .contiguous()를 붙여서 메모리를 일렬로 정렬합니다.
                batch_stim = stim_flat[i : end_idx].contiguous().to(self.device)
                
                with torch.cuda.amp.autocast():
                    batch_pred_norm = self.model(batch_stim)
                
                batch_pred_raw = batch_pred_norm.float() * self.resp_std + self.resp_mean
                
                pred_resp_flat[i : end_idx] = batch_pred_raw
            
            # 🌟 [수정 2] view로 형태를 바꾸기 전에도 메모리 정렬 상태를 확실히 보장합니다.
            pred_resp = pred_resp_flat.contiguous().view(n_variants, n_trs, -1)
            
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

    print(f"[*] Starting Multiplex-GAT (M-GAT) Decoding for {args.subject} / {args.task}...")

    # Load Responses
    resp_path = os.path.join(config.DATA_TEST_DIR, "test_response", args.subject, args.experiment, args.task + ".hf5")
    with h5py.File(resp_path, "r") as hf:
        full_resp = np.nan_to_num(hf["data"][:])
    
    # Load GPT & LM
    with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)
        
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "model"), vocab=gpt_vocab, device=config.GPT_DEVICE)
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass=config.LM_MASS, nuc_ratio=config.LM_RATIO)

    # Load Baseline Models 
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle=True)
    
    baseline_em = np.load(os.path.join(load_location, "encoding_model_%s.npz" % gpt_checkpoint))
    noise_model = baseline_em["noise_model"]
    
    # [수정됨] Load Trained M-GAT Model
    mgat_path = os.path.join(load_location, f"mgat_func_{gpt_checkpoint}.pt")
    print(f"[*] Loading M-GAT from {mgat_path}")
    checkpoint = torch.load(mgat_path, weights_only=False)
    
    tr_stats = checkpoint["tr_stats"]
    word_stats = checkpoint["word_stats"]
    voxels = checkpoint["voxels"]
    resp_mean = checkpoint["resp_mean"]
    resp_std = checkpoint["resp_std"]
    hidden_dim = checkpoint["hidden_dim"]
    n_heads = checkpoint.get("n_heads", 4)
    
    # [수정됨] Load Dual Adjacency Matrices
    print("[*] Loading Dual Graphs (Functional & ROI)...")
    graph_data_func = np.load(os.path.join(load_location, "adjacency_matrix.npz"))
    adj_func = graph_data_func['adj']
    
    graph_data_roi = np.load(os.path.join(load_location, "refined_roi_adjacency_matrix.npz"))
    adj_roi = graph_data_roi['adj']
    
    input_dim = tr_stats[0].shape[0] * len(config.STIM_DELAYS)
    num_voxels = len(voxels)
    
    # [수정됨] Initialize M-GAT Model
    mgat_model = MGATModel(
        input_dim, num_voxels, adj_func, adj_roi, hidden_dim=hidden_dim, n_heads=n_heads
    ).to(config.EM_DEVICE)
    mgat_model.load_state_dict(checkpoint["model_state_dict"])
    
    # Subset responses
    gat_resp = full_resp[:, voxels]
    
    # Wrapper
    em = MGATEncodingModel(mgat_model, gat_resp, noise_model, resp_mean, resp_std, device=config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    
    # Decode
    print("[*] Predicting Word Rates...")
    word_rate = predict_word_rate(full_resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
    starttime = -10 if args.experiment == "perceived_speech" else 0
    word_times, tr_times = predict_word_times(word_rate, full_resp, starttime=starttime)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)

    print(f"[*] Setting Beam Width to 200 (No Compromise Mode)")
    decoder = Decoder(word_times, 200) 
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device=config.SM_DEVICE)
    
    print("[*] Decoding... (M-GAT Inference)")

    with torch.no_grad():
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
            
                likelihoods = em.prs(stim, trs)
            
                local_extensions = [Hypothesis(parent=hyp, extension=x) for x in zip(nuc, logprobs, extend_embs)]
                decoder.add_extensions(local_extensions, likelihoods, nextensions)

                # RAM 방어
                del stim
                del extend_embs
                del extend_words
                if c % 50 == 0:
                    gc.collect()
            
            decoder.extend(verbose=False)
        
            if sample_index % 10 == 0:
                gc.collect()
        
            if sample_index % 10 == 0:
                if decoder.beam and decoder.beam[0].words:
                    print(f"Step {sample_index}/{len(word_times)}: ...{' '.join(decoder.beam[0].words[-5:])} [Sync & Safe]")
                else:
                    print(f"Step {sample_index}/{len(word_times)}: [Start] [Sync & Safe]")

    # Save Results (기존 GAT 결과에 덮어쓰지 않도록 파일명 변경)
    save_location = os.path.join(config.RESULT_DIR, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok=True)
    save_path = os.path.join(save_location, args.task + "_MGAT")
    decoder.save(save_path)
    print(f"[*] Multiplex-GAT Decoding results saved to {save_path}.npz")