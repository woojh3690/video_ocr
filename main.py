import os
import cv2
import numpy as np
import pandas as pd
from thefuzz import fuzz
from PIL import Image
import torch

from transformers import AutoModel, AutoTokenizer

# OCR 결과를 저장할 파일 경로
ocr_results_file = 'ocr_results.pkl'  # 또는 'ocr_results.csv'

# OCR 결과 파일이 존재하면 로드하고 OCR 단계를 건너뜁니다.
if os.path.exists(ocr_results_file):
    print("OCR 결과 파일이 존재합니다. OCR 단계를 건너뜁니다.")
    with open(ocr_results_file, 'rb') as f:
        data = pd.read_pickle(f)
        df = data['ocr_results']
        frame_rate = data['frame_rate']
        total_frames = data['total_frames']
else:
    # CUDA 디바이스가 있는지 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)

    # 모델 로드
    model = AutoModel.from_pretrained(
        'ucaslcl/GOT-OCR2_0',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device,
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # 모델 평가 모드 및 디바이스로 이동
    if device == 'cuda':
        model = model.eval().cuda()
    else:
        model = model.eval().to(device)

    # 비디오 파일 열기
    video_path = 'test.mp4'  # 여기에 비디오 파일 경로를 입력하세요
    cap = cv2.VideoCapture(video_path)

    # 비디오 열기에 실패한 경우 처리
    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        exit()

    # 영상 메타 데이터
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 처리할 구간 설정 (초 단위)
    start_time = 60.0 * 2 + 12.0    # 시작 시간 (초)
    end_time = 60.0 * 2 + 19.0      # 종료 시간 (초)

    # 시작 프레임과 종료 프레임 계산
    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)

    # 시작 프레임이 총 프레임 수보다 큰 경우 처리
    if start_frame >= total_frames:
        print("시작 시간이 비디오의 길이를 초과합니다.")
        cap.release()
        exit()

    # 종료 프레임이 총 프레임 수보다 큰 경우 총 프레임 수로 설정
    if end_frame > total_frames:
        end_frame = total_frames - 1

    # OCR 결과를 저장할 리스트 초기화
    ocr_results = []
    
    # 비디오 캡처를 시작 프레임으로 설정
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 프레임 번호를 시작 프레임으로 설정
    frame_number = start_frame

    # 비디오의 끝까지 프레임을 읽음
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # 현재 프레임 번호가 종료 프레임을 넘으면 중지
            if frame_number > end_frame:
                break

            # OpenCV의 BGR 이미지를 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # NumPy 배열을 PIL 이미지로 변환
            image = Image.fromarray(frame_rgb)
            # 여기서 OCR을 수행하세요
            ocr_text = model.chat(tokenizer, image, ocr_type='ocr', gradio_input=True)
            # OCR 결과를 리스트에 추가
            ocr_results.append({'frame_number': frame_number, 'text': ocr_text})
            frame_number += 1
        else:
            break

    # 비디오 캡처 객체 해제
    cap.release()

    # OCR 결과를 판다스 데이터프레임으로 변환
    df = pd.DataFrame(ocr_results)

    # 인덱스를 프레임 번호로 설정
    df.set_index('frame_number', inplace=True)

    # OCR 결과와 메타데이터를 함께 저장
    with open(ocr_results_file, 'wb') as f:
        pd.to_pickle({'ocr_results': df, 'frame_rate': frame_rate, 'total_frames': total_frames}, f)

    # OCR 단계가 끝났으므로 비디오 캡처 객체 해제
    cap.release()

# 2. 유사한 텍스트 그룹화
# 자막 그룹 ID를 저장할 열 생성
df['group_id'] = np.nan

# 이미 그룹화된 인덱스를 저장할 집합
grouped_indices = set()

group_id = 0
for idx, row in df.iterrows():
    current_text = row['text']
    similar_indices = []

    print(current_text)

    if idx in grouped_indices:
        continue  # 이미 그룹화된 경우

    # 현재 프레임 이후의 데이터와 비교
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

    # 그룹화
    indices = [idx] + similar_indices
    df.loc[indices, 'group_id'] = group_id
    grouped_indices.update(indices)
    group_id += 1

# 그룹 ID를 정수형으로 변환
df['group_id'] = df['group_id'].astype(int)

# 3. 자막의 시작 시간과 종료 시간 결정
subtitles = []

for gid, group in df.groupby('group_id'):
    start_frame_group = group.index.min()
    end_frame_group = group.index.max()
    start_time_sub = start_frame_group / frame_rate
    end_time_sub = (end_frame_group + 1) / frame_rate  # 프레임이 0부터 시작한다고 가정

    # 최종 자막 텍스트 결정 (가장 빈도수가 높은 텍스트 선택)
    final_text = group['text'].mode()[0]
    if final_text == "":
        continue

    subtitles.append({
        'index': gid + 1,
        'start_time': start_time_sub,
        'end_time': end_time_sub,
        'text': final_text
    })

# 4. SRT 파일 생성
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

with open('test.srt', 'w', encoding='utf-8') as f:
    for subtitle in subtitles:
        f.write(f"{subtitle['index']}\n")
        start = format_time(subtitle['start_time'])
        end = format_time(subtitle['end_time'])
        f.write(f"{start} --> {end}\n")
        f.write(f"{subtitle['text']}\n\n")

print("SRT 자막 파일이 생성되었습니다: test.srt")
