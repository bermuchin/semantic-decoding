import numpy as np

# 저장된 스코어 로드
path = r'C:\Users\bumjinkim\Desktop\semantic-decoding\scores\S3\perceived_speech\wheretheressmoke_GCN_func.npz'
data = np.load(path, allow_pickle=True)

# 스토리 전체 점수 확인
story_scores = data['story_scores'].item()
print("\n[ Phase 3 GCN Decoding Results ]")
for metric, val in story_scores.items():
    print(f" - {metric[1]}: {val.mean():.4f}")

# Z-score 확인 (통계적 유의성)
story_z = data['story_zscores'].item()
print(f" - BERT Z-score: {story_z[(args.task if 'args' in locals() else 'wheretheressmoke', 'BERT')]:.4f}")