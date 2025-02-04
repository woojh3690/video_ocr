import os
import sys
import re
import shutil
import uuid
import asyncio
import time
import pickle
import atexit
import signal
from typing import Dict, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import aiofiles

from core.ocr import process_ocr


app = FastAPI()

# 업로드된 비디오 파일 저장 경로
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# 글로벌 작업 상태 저장소
# 예: { task_id: { "video_filename": str, "status": "running"/"completed"/"failed",
#                   "progress": 0~100, "messages": [progress update objects],
#                   "result": srt 파일 경로, "error": str, "start_time": timestamp } }
PICKLE_FILENAME = 'tasks.pkl'
tasks: Dict[str, Dict] = {}

# 클라이언트 WebSocket 연결 (클라이언트당 하나의 WebSocket)
global_websocket_connections: List[WebSocket] = []

def load_tasks():
    global tasks
    if os.path.exists(PICKLE_FILENAME):
        try:
            with open(PICKLE_FILENAME, 'rb') as f:
                tasks = pickle.load(f)
            print(f"{PICKLE_FILENAME}에서 tasks 로드 성공")
        except Exception as e:
            print("tasks 로드 중 오류 발생:", e)
            tasks = {}
    else:
        tasks = {}

def save_tasks():
    try:
        with open(PICKLE_FILENAME, 'wb') as f:
            pickle.dump(tasks, f)
        print(f"tasks를 {PICKLE_FILENAME}에 저장했습니다.")
    except Exception as e:
        print("tasks 저장 중 오류 발생:", e)

# atexit를 사용해 프로그램 종료 시 자동 저장
atexit.register(save_tasks)

def handle_termination(signum, frame):
    print(f"종료 시그널({signum}) 수신 - tasks 저장 중...")
    save_tasks()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_termination)
signal.signal(signal.SIGINT, handle_termination)

# 프로그램 시작 시 tasks 로드
load_tasks()

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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


# ---------------------------
# 백그라운드 OCR 작업 관련 함수 및 엔드포인트
# ---------------------------
@app.post("/start_ocr/")
async def start_ocr_endpoint(
    video_filename: str = Form(...), 
    x: int = Form(...), 
    y: int = Form(...), 
    width: int = Form(...), 
    height: int = Form(...), 
    interval_value: float = Form(...)
):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "task_id": task_id,
        "video_filename": video_filename,
        "status": "running",
        "progress": 0,
        "estimated_completion": "TDB",
        "messages": [],
        "start_time": time.time(),
        "cancelled": False  # 취소 플래그 추가
    }
    
    # 백그라운드에서 OCR 작업 실행
    asyncio.create_task(run_ocr_task(task_id, video_filename, x, y, width, height, interval_value))
    return {"task_id": task_id}


async def run_ocr_task(task_id, video_filename, x, y, width, height, interval):
    task = tasks[task_id]
    try:
        async for progress in process_ocr(video_filename, x, y, width, height, interval):
            # 취소 요청이 들어왔는지 확인
            if task.get("cancelled"):
                task["status"] = "cancelled"
                await broadcast_update(task)
                return  # 작업 중지

            try:
                task["progress"] = progress
                task["estimated_completion"] = calculate_estimated_completion(task)
                await broadcast_update(task)
            except Exception as e:
                print("Progress message 처리 중 에러:", e)
        # OCR 작업 정상 완료 시
        task["status"] = "completed"
        filename_without_ext = os.path.splitext(os.path.basename(video_filename))[0]
        srt_path = os.path.join(UPLOAD_DIR, f"{filename_without_ext}.srt")
        task["result"] = srt_path if os.path.exists(srt_path) else None
        await broadcast_update(task)
    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        print("OCR 작업 중 에러:", e)
        await broadcast_update(task)


@app.post("/cancel_ocr/")
async def cancel_ocr(task_id: str = Form(...)):
    if task_id in tasks:
        tasks[task_id]["cancelled"] = True
        tasks[task_id]["status"] = "cancelling"
        await broadcast_update(tasks[task_id])
        return {"detail": f"Task {task_id} 취소 요청이 접수되었습니다."}
    else:
        raise HTTPException(status_code=404, detail="Task not found")
    

async def broadcast_update(message: dict):
    """
    모든 연결된 클라이언트 WebSocket에 message(JSON)를 전송합니다.
    전송에 실패한 경우 해당 WebSocket은 리스트에서 제거합니다.
    """
    to_remove = []
    for ws in global_websocket_connections:
        try:
            await ws.send_json(message)
        except Exception as e:
            print("WebSocket 전송 에러:", e)
            to_remove.append(ws)
    for ws in to_remove:
        if ws in global_websocket_connections:
            global_websocket_connections.remove(ws)


# ---------------------------
# 작업 상태 조회 엔드포인트
# ---------------------------
@app.get("/tasks/")
async def get_tasks():
    return tasks


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


# ---------------------------
# 작업 삭제 관련 함수 및 엔드포인트
# ---------------------------
def delete_task(task_id: str):
    """
    작업 삭제 기능 구현:
    - tasks 딕셔너리에서 해당 작업 삭제
    - 관련 파일(CSV, SRT)이 존재하면 삭제
    """
    if task_id in tasks:
        video_filename = tasks[task_id].get("video_filename")
        if video_filename:
            filename_without_ext = os.path.splitext(os.path.basename(video_filename))[0]
            csv_path = os.path.join(UPLOAD_DIR, f"{filename_without_ext}.csv")
            srt_path = os.path.join(UPLOAD_DIR, f"{filename_without_ext}.srt")
            # 파일이 존재하면 삭제
            if os.path.exists(csv_path):
                os.remove(csv_path)
            if os.path.exists(srt_path):
                os.remove(srt_path)
        del tasks[task_id]


@app.delete("/tasks/{task_id}")
async def delete_task_endpoint(task_id: str):
    delete_task(task_id)
    return {"detail": f"Task {task_id} 삭제 요청이 처리되었습니다."}


# ---------------------------
# 예상 완료 시간 계산 함수 구현
# ---------------------------
def calculate_estimated_completion(task: Dict) -> str:
    """
    작업 정보를 바탕으로 작업 완료까지 남은 시간을 계산하여 HH:mm:ss 단위로 반환합니다.
    진행률이 0이면 TDB 문자열을 반환합니다.
    """
    progress = task.get("progress", 0)
    start_time = task.get("start_time")
    if not start_time or progress <= 0:
        return "TDB"
    elif progress == 100:
        return "00:00:00"
    elapsed = time.time() - start_time
    remaining = elapsed * (100 - progress) / progress
    return time.strftime("%H:%M:%S", time.gmtime(remaining))


# ---------------------------
# WebSocket 엔드포인트: 클라이언트당 하나의 WebSocket 연결
# ---------------------------
@app.websocket("/ws/tasks")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global_websocket_connections.append(websocket)
    try:
        while True:
            # 클라이언트로부터 받은 메시지를 처리하거나,
            # 단순히 연결 유지용으로 대기
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in global_websocket_connections:
            global_websocket_connections.remove(websocket)

@app.get("/download_srt/{video_filename}")
async def download_srt(video_filename: str):
    video_filename = os.path.splitext(os.path.basename(video_filename))[0]
    subtitle_name = f"{video_filename}.srt"
    srt_file = os.path.join(UPLOAD_DIR, subtitle_name)
    if os.path.exists(srt_file):
        return FileResponse(srt_file, media_type='application/octet-stream', filename=subtitle_name)
    else:
        return {"error": "SRT file not found"}
