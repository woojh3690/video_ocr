import os
import sys
import re
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import aiofiles

from core.ocr import process_ocr


app = FastAPI()

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 업로드된 비디오 파일 저장 경로
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    # 동일한 파일명이 존재하는 경우 업로드 스킵
    if os.path.exists(file_location):
        return {"filename": file.filename, "status": "skipped", "message": "File already exists."}
    
    # 파일 저장
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": file.filename, "status": "uploaded"}

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
async def start_ocr(video_filename: str = Form(...), 
                    x: int = Form(...), y: int = Form(...), 
                    width: int = Form(...), height: int = Form(...)):
    generator = process_ocr(video_filename, x, y, width, height)
    return StreamingResponse(generator, media_type="text/event-stream")

@app.get("/download_srt/{video_filename}")
async def download_srt(video_filename: str):
    video_filename = os.path.splitext(os.path.basename(video_filename))[0]
    subtitle_name = f"{video_filename}.srt"
    srt_file = f"./uploads/{subtitle_name}"
    if os.path.exists(srt_file):
        return FileResponse(srt_file, media_type='application/octet-stream', filename=subtitle_name)
    else:
        return {"error": "SRT file not found"}
