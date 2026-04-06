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

# 작성하신 파일에서 GAT 모델 클래스를 가져옵니다.
try:
    from train_GAT_func import GATModel
except ImportError:
    raise ImportError("train_GAT_func.py 파일이 같은 폴더에 있어야 합니다.")

# --- GAT Encoding Model Wrapper ---
class GATEncodingModel:
    def __init__(self, model, resp, noise_model, resp_mean, resp_std, device="cuda"):
        self.model = model
        self.model.eval()
        self.resp = torch.from_numpy(resp).float().to(device)
        self.device = device
        self.sigma = noise_model
        
        self.resp_mean = torch.from_numpy(resp_mean).float().to(device)
        self.resp_std = torch.from_numpy(resp_std).float().to(device)
        
        self.precision = None
        # [VRAM 방어] H100 환경을 고려하여 배치 사이즈 상향 조정 가능 (예: 16~32)
        self.inference_batch_size = 4

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
        
        pred_resp_list = []
        
        with torch.no_grad():
            for i in range(0, total_samples, self.inference_batch_size):
                batch_stim = stim_flat[i : i + self.inference_batch_size].to(self.device)
                
                # GAT 추론 및 Z-score 복원
                batch_pred_norm = self.model(batch_stim)
                batch_pred_raw = batch_pred_norm * self.resp_std + self.resp_mean
                
                pred_resp_list.append(batch_pred_raw)
            
            pred_resp_flat = torch.cat(pred_resp_list, dim=0)
            pred_resp = pred_resp_flat.view(n_variants, n_trs, -1)
            
            # 우도 계산
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

    print(f"[*] Starting func GAT Decoding for {args.subject} / {args.task}...")

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
    
    # Load Trained GAT Model
    gat_path = os.path.join(load_location, f"gat_func_{gpt_checkpoint}.pt")
    print(f"[*] Loading func GAT from {gat_path}")
    checkpoint = torch.load(gat_path, weights_only=False)
    
    tr_stats = checkpoint["tr_stats"]
    word_stats = checkpoint["word_stats"]
    voxels = checkpoint["voxels"]
    resp_mean = checkpoint["resp_mean"]
    resp_std = checkpoint["resp_std"]
    hidden_dim = checkpoint["hidden_dim"]
    n_heads = checkpoint.get("n_heads", 4)
    
    # Load Adjacency Matrix
    graph_data = np.load(os.path.join(load_location, "adjacency_matrix.npz"))
    adj_matrix = graph_data['adj']
    
    input_dim = tr_stats[0].shape[0] * len(config.STIM_DELAYS)
    num_voxels = len(voxels)
    
    # Initialize GAT Model
    gat_model = GATModel(
        input_dim, num_voxels, adj_matrix, hidden_dim=hidden_dim, n_heads=n_heads
    ).to(config.EM_DEVICE)
    gat_model.load_state_dict(checkpoint["model_state_dict"])
    
    # Subset responses
    gat_resp = full_resp[:, voxels]
    
    # Wrapper
    em = GATEncodingModel(gat_model, gat_resp, noise_model, resp_mean, resp_std, device=config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    
    # Decode
    print("[*] Predicting Word Rates...")
    word_rate = predict_word_rate(full_resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
    starttime = -10 if args.experiment == "perceived_speech" else 0
    word_times, tr_times = predict_word_times(word_rate, full_resp, starttime=starttime)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)

    # 빔 폭 설정
    print(f"[*] Setting Beam Width to 200 (No Compromise Mode)")
    decoder = Decoder(word_times, 200) 
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device=config.SM_DEVICE)
    
    print("[*] Decoding... (func GAT Inference)")

    # --- 최적화된 디코딩 루프 ---
    # --- 최적화된 하이브리드 배치 디코딩 루프 ---
    with torch.no_grad():
        for sample_index in range(len(word_times)):
            trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
            ncontext = decoder.time_window(sample_index, config.LM_TIME, floor=5)
            beam_nucs = lm.beam_propose(decoder.beam, ncontext)
            
            # 가설들을 적정 단위(예: 20개)로 나누어 처리합니다.
            sub_batch_size = 20 
            all_hyps = list(decoder.get_hypotheses())
            
            for i in range(0, len(all_hyps), sub_batch_size):
                sub_hyps = all_hyps[i : i + sub_batch_size]
                
                sub_stims = []
                sub_metadata = []
                
                # 1. 미니 배치만큼만 데이터 수집
                for c_idx, (hyp, nextensions) in enumerate(sub_hyps):
                    actual_idx = i + c_idx
                    nuc, logprobs = beam_nucs[actual_idx]
                    if len(nuc) < 1: continue
                    
                    extend_words = [hyp.words + [x] for x in nuc]
                    extend_embs = list(features.extend(extend_words))
                    
                    stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
                    sub_stims.append(stim)
                    sub_metadata.append((hyp, nuc, logprobs, extend_embs, nextensions))
                
                # 2. 수집된 미니 배치를 한 번에 연산
                if sub_stims:
                    # CPU에서의 cat 연산 규모를 줄여 병목 방지
                    big_sub_stim = torch.cat(sub_stims, dim=0) 
                    sub_likelihoods = em.prs(big_sub_stim, trs) 
                    
                    # 3. 결과 배분
                    curr_pos = 0
                    for hyp, nuc, logprobs, extend_embs, nextensions in sub_metadata:
                        n_vars = len(nuc)
                        likes = sub_likelihoods[curr_pos : curr_pos + n_vars]
                        
                        local_exts = [Hypothesis(parent=hyp, extension=x) 
                                     for x in zip(nuc, logprobs, extend_embs)]
                        decoder.add_extensions(local_exts, likes, nextensions)
                        curr_pos += n_vars
                
                # 미니 배치 처리 후 즉시 메모리 해제
                del sub_stims
                del sub_metadata
                gc.collect()

            # 스텝 종료 후 빔 확장
            decoder.extend(verbose=False)
            
            # 10 스텝마다 GPU 캐시 완전 비우기
            if sample_index % 10 == 0:
                torch.cuda.empty_cache()
                if decoder.beam and decoder.beam[0].words:
                    print(f"Step {sample_index}/{len(word_times)}: {' '.join(decoder.beam[0].words[-5:])}")

    # 최종 결과 저장
    save_location = os.path.join(config.RESULT_DIR, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok=True)
    save_path = os.path.join(save_location, args.task + "_GAT_func")
    decoder.save(save_path)
    print(f"[*] func GAT Decoding results saved to {save_path}.npz")