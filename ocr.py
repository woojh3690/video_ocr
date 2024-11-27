import os
import csv

import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from merging_module import merge_ocr_texts  # 모듈 임포트

system_prompt = 'Extract all the subtitles and text from image. \
If there is no visible text in this image then output: None'

model_id = "openbmb/MiniCPM-V-2_6-int4"
model = AutoModel.from_pretrained(
    model_id, 
    trust_remote_code=True,
    attn_implementation='flash_attention_2', # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 진행 상황과 OCR 결과 저장
progress = {}
ocr_text_data = []

# few-shot 예제 로딩
few_shot_data = []
i = 0
with open('./few_shot/answer.txt', 'r', encoding="utf-8") as file:
    for line in file:
        shot_image = Image.open(f'./few_shot/shot{i}.jpg').convert('RGB')
        answer = line.strip()
        few_shot_data.append({'role': 'user', 'content': [shot_image, system_prompt]})
        few_shot_data.append({'role': 'assistant', 'content': [answer]})
        i += 1

def do_ocr(images) -> str:
    msgs = []
    for image in images:
        msg = few_shot_data[:]
        msg.append({'role': 'user', 'content': [image, system_prompt]})
        msgs.append(msg)

    responses = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
    )

    return responses

# 동영상 파일에서 프레임을 배치 단위로 생성하는 제너레이터.
def frame_batch_generator(cap: cv2.VideoCapture, last_frame_number: int, batch_size=8):
    last_frame_number = -1 if last_frame_number == None else last_frame_number

    interval = 0.3  # 0.3초마다 OCR 수행
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(frame_rate * interval)))  # 최소 1프레임

    frame_number = -1
    batch = []
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break
        frame_number += 1

        # 처리된 않은 프레임 부터 OCR 진행
        if frame_number <= last_frame_number:
            continue
        
        # 0.3초마다 OCR 수행
        if frame_number % frame_interval != 0:
            continue

        # 이미지 크롭
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        # cropped_img = image.crop((x, y, x+width, y+height))

        # 배치 추가
        batch.append([frame_number, image])

        # 배치가 꽉 찼다면 반환
        if len(batch) >= batch_size:
            yield batch
            batch = []
    # 남은 프레임 반환
    yield batch

def get_last_processed_frame_number(csv_path):
    # 기존 OCR 데이터를 로드하여 진행 상황을 파악
    last_processed_frame_number = None
    if os.path.exists(csv_path):
        last_processed_frame_number = -1
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                frame_number = int(row['frame_number'])
                last_processed_frame_number = max(last_processed_frame_number, frame_number)
                ocr_text_data.append({
                    'frame_number': frame_number,
                    'time': float(row['time']),
                    'text': row['text'],
                })
    return last_processed_frame_number

def process_ocr(video_filename, x, y, width, height):
    UPLOAD_DIR = "uploads"
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    csv_path = os.path.join(UPLOAD_DIR, f"{video_filename}.csv")
    srt_path = os.path.join(UPLOAD_DIR, f"{video_filename}.srt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 영상 정보 추출
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # CSV 파일을 열어둔 채로 진행
    last_processed_frame_number = get_last_processed_frame_number(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['frame_number', 'time', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if last_processed_frame_number == None:
            writer.writeheader()

        for frames in frame_batch_generator(cap, last_processed_frame_number):
            images = [frame[1] for frame in frames]

            # OCR 수행
            ocr_texts = do_ocr(images)

            for i, ocr_text in enumerate(ocr_texts):
                # 후처리 
                if ocr_text == "None" or ocr_text == "(None)" or "image does not contain any" in ocr_text:
                    ocr_text = ""
                ocr_text = ocr_text.replace('\n', '\\n')    # 줄바꿈 문자 이스케이프 처리

                # CSV 파일에 쓰기
                fram_number = frames[i][0]
                entry = {
                    'frame_number': fram_number, 
                    'time': round(fram_number / frame_rate, 3), 
                    'text': ocr_text
                }
                ocr_text_data.append(entry)
                writer.writerow(entry)
            csvfile.flush()  # 버퍼를 즉시 파일에 씁니다.

            # 진행 상황 업데이트
            progress["value"] = (frames[-1][0] / total_frames) * 100

    # 텍스트 병합 모듈 사용
    ocr_progress_data = merge_ocr_texts(ocr_text_data)

    # SRT 파일 생성
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    # 자막 생성
    with open(srt_path, 'w', encoding='utf-8') as f:
        for idx, subtitle in enumerate(ocr_progress_data, start=1):
            start = format_time(subtitle['start_time'])
            end = format_time(subtitle['end_time'])
            subtitle_line = subtitle['text'].replace("\\n", "\n")
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{subtitle_line}\n\n")

    progress["value"] = 100