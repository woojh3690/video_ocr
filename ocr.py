import os
import cv2
import numpy as np
import pandas as pd
from thefuzz import fuzz
from PIL import Image
import torch

from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

# 진행 상황을 저장하는 딕셔너리
progress = {}

# 모델 및 토크나이저 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model: Qwen2ForCausalLM = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map=device,
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id
)
model = model.eval().to(device)

def process_ocr(video_filename, x, y, width, height):
    UPLOAD_DIR = "uploads"
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 전체 프레임 처리
    start_frame = 0
    end_frame = total_frames - 1

    ocr_results = []
    frame_number = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        frame_number += 1
        if ret:
            if frame_number % 24 != 0:
                continue

            if frame_number > end_frame:
                break

            # BGR에서 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # OCR 수행
            ocr_box = f'[{x},{y},{x+width},{y+height}]'
            ocr_text = model.chat(tokenizer, image, ocr_type='ocr', ocr_box=ocr_box, gradio_input=True)

            ocr_results.append({'frame_number': frame_number, 'text': ocr_text})

            # 진행 상황 업데이트
            progress["value"] = int((frame_number / total_frames) * 100)
        else:
            break

    cap.release()

    # 데이터프레임 생성
    df = pd.DataFrame(ocr_results)
    df.set_index('frame_number', inplace=True)

    # 유사한 텍스트 그룹화
    df['group_id'] = np.nan
    grouped_indices = set()
    group_id = 0
    for idx, row in df.iterrows():
        current_text = row['text']
        similar_indices = []

        if idx in grouped_indices:
            continue

        for idx2, row2 in df.loc[idx+1:].iterrows():
            current_text2 = row2['text']

            if idx2 in grouped_indices:
                continue

            if current_text == "":
                if current_text2 == "":
                    similar_indices.append(idx2)
                else:
                    break
            else:
                if current_text2 == "":
                    break
                distance = fuzz.partial_ratio(current_text, current_text2)
                if distance > 50:
                    current_text = current_text2
                    similar_indices.append(idx2)
                else:
                    break

        indices = [idx] + similar_indices
        df.loc[indices, 'group_id'] = group_id
        grouped_indices.update(indices)
        group_id += 1

    df['group_id'] = df['group_id'].astype(int)

    # 자막 생성
    subtitles = []

    for gid, group in df.groupby('group_id'):
        start_frame_group = group.index.min()
        end_frame_group = group.index.max()
        start_time_sub = start_frame_group / frame_rate
        end_time_sub = (end_frame_group + 1) / frame_rate

        final_text = group['text'].mode()[0]
        if final_text == "":
            continue

        subtitles.append({
            'index': gid + 1,
            'start_time': start_time_sub,
            'end_time': end_time_sub,
            'text': final_text
        })

    # SRT 파일 생성
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    with open('output.srt', 'w', encoding='utf-8') as f:
        for subtitle in subtitles:
            f.write(f"{subtitle['index']}\n")
            start = format_time(subtitle['start_time'])
            end = format_time(subtitle['end_time'])
            f.write(f"{start} --> {end}\n")
            f.write(f"{subtitle['text']}\n\n")

    print("SRT 자막 파일이 생성되었습니다: output.srt")
    progress["value"] = 100  # 진행 상황 완료로 업데이트
