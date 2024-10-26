import os
import cv2
from PIL import Image
import torch
from thefuzz import fuzz
from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
import csv

# 진행 상황과 OCR 결과를 저장하는 전역 변수 및 락
progress = {}
ocr_progress_data = []

ocr_text_data = []

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

    frame_number = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    current_subtitle = None

    while cap.isOpened():
        ret, frame = cap.read()
        frame_number += 1
        if ret:
            if frame_number % 24 != 0:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            ocr_box = f'[{x},{y},{x+width},{y+height}]'
            ocr_text = model.chat(tokenizer, image, ocr_type='ocr', ocr_box=ocr_box, gradio_input=True)

            # ocr_text 데이터를 저장
            current_time = frame_number / frame_rate
            ocr_text_data.append({'time': frame_number / frame_rate, 'text': ocr_text})

            current_time = frame_number / frame_rate

            if current_subtitle is None:
                current_subtitle = {
                    'start_time': current_time,
                    'end_time': current_time,
                    'text': ocr_text
                }
            else:
                similarity = fuzz.partial_ratio(current_subtitle['text'], ocr_text)
                if similarity > 50:
                    current_subtitle['end_time'] = current_time
                else:
                    ocr_progress_data.append(current_subtitle)
                    current_subtitle = {
                        'start_time': current_time,
                        'end_time': current_time,
                        'text': ocr_text
                    }

            progress["value"] = int((frame_number / total_frames) * 100)
        else:
            break

    cap.release()

    if current_subtitle:
        ocr_progress_data.append(current_subtitle)

    # SRT 파일 생성
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    with open('./uploads/output.srt', 'w', encoding='utf-8') as f:
        for idx, subtitle in enumerate(ocr_progress_data):
            idx = idx+1
            start = format_time(subtitle['start_time'])
            end = format_time(subtitle['end_time'])
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{subtitle['text']}\n\n")

    # CSV 파일 생성 (전체 OCR 텍스트 데이터 저장)
    with open('./uploads/ocr_text_output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['time', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in ocr_text_data:
            writer.writerow(entry)

    # CSV 파일 생성 (자막 데이터 저장)
    with open('./uploads/output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['start_time', 'end_time', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for subtitle in ocr_progress_data:
            writer.writerow({
                'start_time': format_time(subtitle['start_time']),
                'end_time': format_time(subtitle['end_time']),
                'text': subtitle['text']
            })

    progress["value"] = 100
