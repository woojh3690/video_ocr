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
ollama_url = os.environ["OLLAMA_URL"]

# 진행 상황과 OCR 결과를 저장하는 전역 변수 및 락
progress = {}
ocr_progress_data = []

ocr_text_data = []

if is_ollama:
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
            messages=[{
                'role': 'user',
                'content': 'Extract all the text from image like {"extract":"ocr_text"}. If there is no text then {"extract":""}',
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
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 프레임 간격 계산
    interval = 0.3  # 0.5초마다 OCR 수행
    frame_interval = max(1, int(round(frame_rate * interval)))  # 최소 1프레임

    while cap.isOpened():
        ret, frame = cap.read()
        frame_number += 1
        if ret:
            if frame_number % frame_interval != 0:
                continue  # 다음 프레임으로 넘어갑니다

            # 이미지 크롭
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            cropped_img = image.crop((x, y, x+width, y+height))

            # OCR 수행
            ocr_text = do_ocr(cropped_img)

            # ocr_text 데이터를 저장
            current_time = frame_number / frame_rate
            ocr_text_data.append({'time': current_time, 'text': ocr_text})

            progress["value"] = int((frame_number / total_frames) * 100)
        else:
            break

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
    with open(f'./uploads/{video_filename}.srt', 'w', encoding='utf-8') as f:
        for idx, subtitle in enumerate(ocr_progress_data, start=1):
            start = format_time(subtitle['start_time'])
            end = format_time(subtitle['end_time'])
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{subtitle['text']}\n\n")

    # CSV 파일 생성 (전체 OCR 텍스트 데이터 저장)
    with open(f'./uploads/{video_filename}.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['time', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in ocr_text_data:
            writer.writerow(entry)

    progress["value"] = 100
