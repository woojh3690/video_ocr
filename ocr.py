import os
import io
import sys
import csv
import json
import base64

import cv2
from PIL import Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler

from merging_module import merge_ocr_texts  # 모듈 임포트

system_prompt = 'You are a ocr assistant. Extract all the subtitles and text in JSON format. \
Group the subtitles according to color of the subtitles; \
subtitles with the same color belong to the same group.\n\
Use the following JSON format: {\"ocr_subtitles_group\":[[\"group1 first subtitle\",\"group1 second subtitle\",\"...\"],\
[\"group2 first subtitle\",\"group2 second subtitle\",\"...\"]]}\n\
If there is no subtitles then: {\"ocr_subtitles_group\":[]}'

class SuppressStdoutStderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )

        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )

        self.outnull_file.close()
        self.errnull_file.close()

with SuppressStdoutStderr():
    chat_handler = MiniCPMv26ChatHandler(clip_model_path="models/mmproj-model-f16.gguf")
    llm = Llama(
        model_path="models/ggml-model-Q8_0.gguf",
        chat_handler=chat_handler,
        n_ctx=16384,
        n_gpu_layers=-1,
        verbose=False,
    )

# 진행 상황과 OCR 결과 저장
progress = {}
ocr_text_data = []

def do_ocr(image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')  # PNG 형식으로 변환
    img_byte_arr.seek(0)  # 시작 위치로 포인터 이동

    # 메모리에 저장된 이미지를 Base64로 인코딩
    image_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
    image_base64 = f"data:image/jpg;base64,{image_base64}"

    with SuppressStdoutStderr():
        response = llm.create_chat_completion(
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user", 
                    "content": [{"type": "image_url", "image_url": {"url": image_base64}}]
                },
            ],
            response_format = {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "ocr_subtitles_group": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        }
                    },
                    "required": [
                        "ocr_subtitles_group"
                    ]
                },
            },
            temperature=0.0,
        )

    content = response["choices"][0]["message"]["content"]
    ocr_subtitles_group = json.loads(content)["ocr_subtitles_group"]

    # 줄 병합
    merged_subtitle = []
    for subtitles in ocr_subtitles_group:
        merged_subtitle.append("\n".join(subtitles))
    result = "\n\n".join(merged_subtitle)
    return result

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

    # 기존 OCR 데이터를 로드하여 진행 상황을 파악합니다.
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

    # CSV 파일을 열어둔 채로 진행합니다.
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['frame_number', 'time', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if last_processed_frame_number == -1:
            writer.writeheader()

        while cap.isOpened():
            ret, frame = cap.read()
            frame_number += 1
            if not ret:
                break

            if frame_number % frame_interval != 0:
                continue  # 다음 프레임으로 넘어갑니다

            # 처리된 않은 프레임 부터 OCR 진행
            if frame_number > last_processed_frame_number:
                # 이미지 크롭
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                cropped_img = image.crop((x, y, x+width, y+height))

                # OCR 수행
                try:
                    ocr_text = do_ocr(cropped_img)
                except Exception as e:
                    print(f"JSON 디코딩 오류 발생: {e}")
                    continue

                # 줄바꿈 문자 이스케이프 처리 
                ocr_text = ocr_text.replace('\n', '\\n')

                # ocr_text 데이터를 저장
                current_time = frame_number / frame_rate
                entry = {'frame_number': frame_number, 'time': current_time, 'text': ocr_text}
                ocr_text_data.append({
                    'time': current_time,
                    'text': ocr_text,  # 원본 텍스트를 저장
                    'frame_number': frame_number
                })
                writer.writerow(entry)
                csvfile.flush()  # 버퍼를 즉시 파일에 씁니다.

            progress["value"] = (frame_number / total_frames) * 100

    cap.release()

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
