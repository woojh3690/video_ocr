import os
import re
import shutil
import threading
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import aiofiles
from pydantic import BaseModel

import cv2
import numpy as np
import pandas as pd
from thefuzz import fuzz
from PIL import Image
import torch

from transformers import AutoModel, AutoTokenizer

app = FastAPI()

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 업로드된 비디오 파일 저장 경로
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# 진행 상황을 저장하는 딕셔너리
progress = {}

# 모델 및 토크나이저 로드 (서버 시작 시 한 번만 로드)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map=device,
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id
)
model = model.eval().to(device)

class OCRRequest(BaseModel):
    video_filename: str
    x: int
    y: int
    width: int
    height: int

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

@app.get("/videos/{video_filename}")
async def get_video(request: Request, video_filename: str):
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = os.path.getsize(video_path)
    headers = {}
    status_code = 200

    range_header = request.headers.get('range')
    if range_header:
        range_match = re.match(r'bytes=(\d+)-(\d+)?', range_header)
        if range_match:
            start = int(range_match.group(1))
            end = range_match.group(2)
            if end:
                end = int(end)
            else:
                end = file_size - 1

            if start >= file_size or end >= file_size:
                raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")

            content_length = end - start + 1
            headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
            headers['Accept-Ranges'] = 'bytes'
            headers['Content-Length'] = str(content_length)
            status_code = 206

            async def stream_video():
                async with aiofiles.open(video_path, 'rb') as f:
                    await f.seek(start)
                    bytes_to_read = content_length
                    chunk_size = 1024 * 1024  # 1MB
                    while bytes_to_read > 0:
                        read_size = min(chunk_size, bytes_to_read)
                        data = await f.read(read_size)
                        if not data:
                            break
                        yield data
                        bytes_to_read -= len(data)

            return StreamingResponse(stream_video(), status_code=status_code, headers=headers, media_type="video/mp4")
    else:
        headers['Accept-Ranges'] = 'bytes'
        return FileResponse(video_path, headers=headers, media_type='video/mp4')


@app.post("/start_ocr/")
async def start_ocr(video_filename: str = Form(...), x: int = Form(...), y: int = Form(...), width: int = Form(...), height: int = Form(...)):
    # OCR 처리를 별도의 스레드에서 실행
    thread = threading.Thread(target=process_ocr, args=(video_filename, x, y, width, height))
    thread.start()
    return {"status": "OCR processing started"}

@app.get("/progress/")
async def get_progress():
    return {"progress": progress.get("value", 0)}

@app.get("/download_srt/")
async def download_srt():
    srt_file = "output.srt"
    if os.path.exists(srt_file):
        return FileResponse(srt_file, media_type='application/octet-stream', filename=srt_file)
    else:
        return {"error": "SRT file not found"}

def process_ocr(video_filename, x, y, width, height):
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
    frame_number = start_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_number > end_frame:
                break

            # 선택된 영역으로 프레임을 크롭
            cropped_frame = frame[y:y+height, x:x+width]

            # BGR에서 RGB로 변환
            frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # OCR 수행
            ocr_text = model.chat(tokenizer, image, ocr_type='ocr', gradio_input=True)

            ocr_results.append({'frame_number': frame_number, 'text': ocr_text})

            frame_number += 1

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
