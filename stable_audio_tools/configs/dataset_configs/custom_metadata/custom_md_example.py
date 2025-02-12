import os
import json
import torch

# 기본 경로 설정
DATASET_BASE_PATH = "/home/solbon1212/datasets/"

# development & evaluation captions 파일 경로
CAPTIONS_PATHS = {
    "development": os.path.join(DATASET_BASE_PATH, "clotho_audio_development/development/prompt.json"),
    "evaluation": os.path.join(DATASET_BASE_PATH, "clotho_audio_evaluation/evaluation/prompt.json"),
}

# JSON 파일을 미리 로드 (파일이 없으면 빈 딕셔너리)
CAPTIONS_DICTS = {}
for key, path in CAPTIONS_PATHS.items():
    if os.path.exists(path):
        with open(path, "r") as f:
            CAPTIONS_DICTS[key] = json.load(f)
    else:
        CAPTIONS_DICTS[key] = {}

def get_custom_metadata(info, audio):
    """오디오 파일별 captions을 JSON에서 로드"""
    
    # 🔍 audio의 타입 확인
    # print(f"🔍 audio type: {type(audio)}, shape: {audio.shape if isinstance(audio, torch.Tensor) else 'N/A'}")

    # 🔹 audio가 Tensor라면 원본 오디오 경로를 사용해야 함
    if isinstance(audio, torch.Tensor):
        if "path" in info:
            audio = info["path"]  # ✅ 올바른 경로 사용
        else:
            raise ValueError(f"🔴 'path' not found in metadata! Check the dataset config.")

    # 🔹 오디오 파일명만 추출
    audio_filename = os.path.basename(audio) if isinstance(audio, str) else "unknown.wav"
    
    # 🔹 captions이 존재하면 반환, 없으면 기본값 설정
    captions = CAPTIONS_DICTS.get(audio_filename, ["No caption available"])
    
    return {"prompt": captions[0]}  # 첫 번째 captions 반환