import os
import io
import cv2
import csv
import json
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
import ollama

from merging_module import merge_ocr_texts  # 모듈 임포트

is_ollama = True if os.environ["OLLAMA"].lower() == "true" else False

system_prompt = 'Extract all the text and subtitle from the image in the format: {"extract":"first_line_subtitle\\nsecond_line_subtitle"}. \
Ensure that line breaks are preserved. \
If there is no text, then return {"extract":""}. \
If there are any double quotation marks within the subtitle text, use escape characters (\\") to preserve them. \
Output format must be JSON'

# 진행 상황과 OCR 결과 저장
progress = {}
ocr_text_data = []

if is_ollama:
    ollama_url = os.environ["OLLAMA_URL"]
    client = ollama.Client(ollama_url)
else:
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

def do_ocr(image) -> str:
    if is_ollama:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')  # PNG 형식으로 변환
        img_byte_arr.seek(0)  # 시작 위치로 포인터 이동

        response = client.chat(
            model='llama3.2-vision:90b',
            messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'images': [img_byte_arr.getvalue()]
            }],
            format="json",
            stream=False,
            options={'temperature': 0}
        )
        response = response['message']['content']
        response = json.loads(response)["extract"]
        return response
    else:
        return model.chat(tokenizer, image, ocr_type='ocr', gradio_input=True)

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
    processed_frames = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                frame_number = int(row['frame_number'])
                processed_frames.add(frame_number)
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
        if not processed_frames:
            writer.writeheader()

        while cap.isOpened():
            ret, frame = cap.read()
            frame_number += 1
            if not ret:
                break

            if frame_number % frame_interval != 0:
                continue  # 다음 프레임으로 넘어갑니다

            if frame_number in processed_frames:
                continue  # 이미 처리된 프레임은 건너뜁니다

            # 이미지 크롭
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            cropped_img = image.crop((x, y, x+width, y+height))

            # OCR 수행
            ocr_text = do_ocr(cropped_img)

            # 줄바꿈 문자 이스케이프 처리
            escaped_text = ocr_text.replace('\n', '\\n')

            # ocr_text 데이터를 저장
            current_time = frame_number / frame_rate
            entry = {'frame_number': frame_number, 'time': current_time, 'text': escaped_text}
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
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{subtitle['text']}\n\n")

    progress["value"] = 100
