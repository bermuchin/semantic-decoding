import os
import numpy as np
import json
import argparse
import config
from utils_eval import generate_null, load_transcript, windows, segment_data, WER, BLEU, METEOR, BERTSCORE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--experiment", type = str, required = True)
    parser.add_argument("--task", type = str, required = True)
    parser.add_argument("--null", type = int, default = 10) # 유의성 검정용
    args = parser.parse_args()
    
    # 평가 구간 정보 로드
    with open(os.path.join(config.DATA_TEST_DIR, "eval_segments.json"), "r") as f:
        eval_segments = json.load(f)
                
    # 성능 지표 로드 (BERTScore 포함)
    metrics = {
        "WER": WER(use_score = True),
        "BLEU": BLEU(n = 1),
        "METEOR": METEOR(),
        "BERT": BERTSCORE(
            idf_sents = np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")), 
            rescale = False, 
            score = "recall"
        )
    }

    # [범진님 전용] GCN 결과 파일 로드 (_GCN_func 접미사)
    pred_path = os.path.join(config.RESULT_DIR, args.subject, args.experiment, f"{args.task}_GCN_func.npz")
    print(f"[*] Loading GCN results from: {pred_path}")
    
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {pred_path}")

    pred_data = np.load(pred_path)
    pred_words, pred_times = pred_data["words"], pred_data["times"]

    # Null 모델 생성 (통계적 유의성 확인용)
    gpt_checkpoint = "perceived"
    print(f"[*] Generating {args.null} Null Models for significance testing...")
    null_word_list = generate_null(pred_times, gpt_checkpoint, args.null)
        
    window_scores, window_zscores = {}, {}
    story_scores, story_zscores = {}, {}

    # 정답 대본(Reference) 로드
    ref_data = load_transcript(args.experiment, args.task)
    ref_words, ref_times = ref_data["words"], ref_data["times"]

    # 20초 윈도우 단위로 데이터 분할
    window_cutoffs = windows(*eval_segments[args.task], config.WINDOW)
    ref_windows = segment_data(ref_words, ref_times, window_cutoffs)
    pred_windows = segment_data(pred_words, pred_times, window_cutoffs)
    null_window_list = [segment_data(null_words, pred_times, window_cutoffs) for null_words in null_word_list]
    
    # 지표별 점수 계산
    for mname, metric in metrics.items():
        print(f"[*] Computing {mname}...")
        window_null_scores = np.array([metric.score(ref = ref_windows, pred = null_windows) 
                                       for null_windows in null_window_list])
        story_null_scores = window_null_scores.mean(1)

        # 실제 점수 및 Z-score 계산
        window_scores[(args.task, mname)] = metric.score(ref = ref_windows, pred = pred_windows)
        window_zscores[(args.task, mname)] = (window_scores[(args.task, mname)] 
                                              - window_null_scores.mean(0)) / (window_null_scores.std(0) + 1e-8)

        story_scores[(args.task, mname)] = window_scores[(args.task, mname)]
        story_zscores[(args.task, mname)] = (story_scores[(args.task, mname)].mean()
                                             - story_null_scores.mean()) / (story_null_scores.std() + 1e-8)
    
    # 결과 저장 (_GCN_func 접미사 유지)
    save_location = os.path.join(config.SCORE_DIR, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok = True)
    save_path = os.path.join(save_location, f"{args.task}_GCN_func.npz")
    
    np.savez(save_path, 
             window_scores = window_scores, window_zscores = window_zscores, 
             story_scores = story_scores, story_zscores = story_zscores)
    
    print(f"[*] Evaluation Complete! Saved to: {save_path}")