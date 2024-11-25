import os
import csv
import threading
import queue

import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from merging_module import merge_ocr_texts  # 모듈 임포트

system_prompt = 'Extract all the subtitles and text from image. \
If there is no visible text in this image then output: None'

model_id = "openbmb/MiniCPM-V-2_6"
model = AutoModel.from_pretrained(
    model_id, 
    trust_remote_code=True,
    attn_implementation='flash_attention_2', # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16
)
model = model.eval().cuda()
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

def do_ocr(image) -> str:
    msgs = few_shot_data[:]
    msgs.append({'role': 'user', 'content': [image, system_prompt]})

    response = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
    )

    if response == "None" or response == "(None)" or "image does not contain any" in response:
        response = ""
    return response

def process_ocr(video_filename, x, y, width, height):
    UPLOAD_DIR = "uploads"
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    csv_path = os.path.join(UPLOAD_DIR, f"{video_filename}.csv")
    srt_path = os.path.join(UPLOAD_DIR, f"{video_filename}.srt")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 기존 OCR 데이터를 로드하여 진행 상황을 파악
    last_processed_frame_number = -1
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                frame_number = int(row['frame_number'])
                last_processed_frame_number = max(last_processed_frame_number, frame_number)
                ocr_text_data.append({
                    'time': float(row['time']),
                    'text': row['text'],
                    'frame_number': frame_number
                })

    frame_number = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 프레임 간격 계산
    interval = 0.3  # 0.3초마다 OCR 수행
    frame_interval = max(1, int(round(frame_rate * interval)))  # 최소 1프레임

    # 작업 큐와 결과 큐 생성
    task_queue = queue.Queue()
    result_queue = queue.Queue()

    # 워커 함수 정의
    def ocr_worker():
        while True:
            task = task_queue.get()
            if task is None: break
            frame_num, image = task
            try:
                ocr_text = do_ocr(image)
                result_queue.put((frame_num, ocr_text))
            except Exception as e:
                print(f"OCR 오류 발생 (프레임 {frame_num}): {e}")
        result_queue.put(None)  # 워커 스레드 종료 신호 보내기

    # 워커 스레드 시작
    worker_thread = threading.Thread(target=ocr_worker)
    worker_thread.start()
    
    # CSV 파일을 열어둔 채로 진행
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['frame_number', 'time', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if last_processed_frame_number == -1:
            writer.writeheader()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_number += 1

            if frame_number % frame_interval != 0:
                continue  # 다음 프레임으로 넘어갑니다

            # 처리되지 않은 프레임부터 OCR 진행
            if frame_number > last_processed_frame_number:
                # 이미지 크롭
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                # cropped_img = image.crop((x, y, x+width, y+height))

                task_queue.put((frame_number, image))   # 작업 큐에 작업 추가
        task_queue.put(None)    # 작업 종료
        cap.release()

        # 모든 작업이 완료될 때까지 결과 처리
        while task_result := result_queue.get():
            result_frame_number, ocr_text = task_result
            current_time = result_frame_number / frame_rate
            entry = {'frame_number': result_frame_number, 'time': current_time, 'text': ocr_text}
            ocr_text_data.append({
                'time': current_time,
                'text': ocr_text,  # 원본 텍스트를 저장
                'frame_number': result_frame_number
            })
            writer.writerow(entry)
            csvfile.flush()  # 버퍼를 즉시 파일에 씁니다.
            progress["value"] = (result_frame_number / total_frames) * 100

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