import pandas as pd
import numpy as np
from thefuzz import fuzz

# 예시 OCR 결과 데이터 생성
# 실제로는 OCR 모델을 통해 얻은 결과를 사용해야 합니다.
# 이 예시에서는 데이터 프레임으로 시뮬레이션합니다.

# 샘플 데이터: 프레임 번호, 추출된 텍스트, 텍스트의 위치 정보(x, y 좌표)
ocr_results = [
    {'frame': 1, 'text': '안녕하세요'},
    {'frame': 2, 'text': '안녕하세요'},
    {'frame': 3, 'text': '안녕하세'},  # OCR 오류
    {'frame': 10, 'text': '저는 AI입니다'},
    {'frame': 11, 'text': '저는 AI입니다'},
    {'frame': 20, 'text': '만나서'},
    {'frame': 21, 'text': '만나서 반갑습니다.'},
    {'frame': 22, 'text': '만나서 반갑습니c.'},  # OCR 오류
    {'frame': 23, 'text': '만나서 반갑습니다. 저는'},
    {'frame': 24, 'text': '만나서 반갑습니다. 저는 우준혁'},
    {'frame': 25, 'text': '만나서 반갑습니다. 저는 우준혁 이라고'},
    {'frame': 26, 'text': '만나서 반갑습니다. 저는 우준혁 이라고 합니다.'},
    {'frame': 29, 'text': '아 그러니?'},
]

df = pd.DataFrame(ocr_results)

# 프레임 레이트 (예: 30fps)
frame_rate = 24.0

# 1. 텍스트 정규화
def normalize_text(text):
    text = text.strip()
    # 필요한 경우 추가 정규화 수행
    return text

df['norm_text'] = df['text'].apply(normalize_text)

# 2. 유사한 텍스트 그룹화
# 자막 그룹 ID를 저장할 열 생성
df['group_id'] = np.nan

group_id = 0
threshold = 5  # Levenshtein 거리 임계값

# 이미 그룹화된 인덱스를 저장할 집합
grouped_indices = set()

for idx, row in df.iterrows():
    if idx in grouped_indices:
        continue  # 이미 그룹화된 경우

    current_text = row['norm_text']
    current_frame = row['frame']
    similar_indices = []

    # 현재 프레임 이후의 데이터와 비교
    for idx2, row2 in df.loc[idx+1:].iterrows():
        current_text2 = row2['norm_text']

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
            if distance > 80:
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
    start_frame = group['frame'].min()
    end_frame = group['frame'].max()
    start_time = start_frame / frame_rate
    end_time = (end_frame + 1) / frame_rate  # 프레임이 0부터 시작한다고 가정

    # 최종 자막 텍스트 결정 (가장 빈도수가 높은 텍스트 선택)
    final_text = max(group['norm_text'], key=len)
    if final_text == "":
        continue

    subtitles.append({
        'index': gid + 1,
        'start_time': start_time,
        'end_time': end_time,
        'text': final_text
    })

# 4. SRT 파일 생성
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
