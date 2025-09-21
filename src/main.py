import os
import sys
import re
import uuid
import asyncio
import time
import pickle
import atexit
import signal
import traceback
from enum import Enum
from json import dumps
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import cv2
import openai
import aiofiles
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from kafka import KafkaProducer
from docker.errors import APIError, DockerException

from core.ocr import process_ocr, UPLOAD_DIR, base_url
from core.docker_manager import DockerManager

class Status(str, Enum):
    waiting = "waiting"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelling = "cancelling"
    cancelled = "cancelled"

@dataclass
class Task:
    task_id: str
    video_filename: str
    status: Status
    progress: int = 0
    estimated_completion: str = "TBD"
    messages: List[dict] = field(default_factory=list)
    task_start_time: Optional[float] = None
    ocr_x: int = 0
    ocr_y: int = 0
    ocr_width: int = 0
    ocr_height: int = 0
    ocr_start_time: Optional[int] = 0
    ocr_end_time: Optional[int] = None
    result: Optional[str] = None
    error: Optional[str] = None

docker_url: str = os.getenv("DOCKER_URL", "tcp://192.168.1.63:2375")
docker_name: str = os.getenv("DOCKER_NAME", "vllm_7b")

producer = KafkaProducer(
    acks=0, 
    compression_type="gzip", 
    bootstrap_servers=[os.getenv("KAFKA_URL", "192.168.1.17:19092")], 
    value_serializer=lambda x: dumps(x).encode("utf-8")
)

app = FastAPI()
docker_manager = DockerManager(docker_url)

# 업로드된 비디오 파일 저장 경로
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# 글로벌 작업 상태 저장소
# 예: { task_id: { "video_filename": str, "status": Status,
#                   "progress": 0~100, "messages": [progress update objects],
#                   "result": srt 파일 경로, "error": str, "task_start_time": timestamp } }
PICKLE_FILENAME = 'tasks.pkl'
tasks: Dict[str, Task] = {}


def get_running_task_id() -> Optional[str]:
    """현재 실행 중인 작업의 ID를 반환합니다."""
    for tid, t in tasks.items():
        if t.status == Status.running:
            return tid
    return None


def get_next_waiting_task_id() -> Optional[str]:
    """대기 중인 다음 작업의 ID를 반환합니다."""
    for tid, t in tasks.items():
        if t.status == Status.waiting:
            return tid
    return None


# 클라이언트 WebSocket 연결 (클라이언트당 하나의 WebSocket)
global_websocket_connections: List[WebSocket] = []

def load_tasks():
    global tasks
    if os.path.exists(PICKLE_FILENAME):
        try:
            with open(PICKLE_FILENAME, 'rb') as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    tasks = {}
                    for tid, data in loaded.items():
                        if isinstance(data, Task):
                            tasks[tid] = data
                        elif isinstance(data, dict):
                            data.pop('interval', None)
                            tasks[tid] = Task(**data)
                        else:
                            tasks[tid] = Task(**{})
                else:
                    tasks = {}
            print(f"{PICKLE_FILENAME}에서 tasks 로드 성공")
        except Exception as e:
            print("tasks 로드 중 오류 발생:", e)
            tasks = {}
    else:
        tasks = {}
    
    # 로드된 테스크 정보에서 실행 중이던 상태를 모두 cancelled 로 변경
    for t in tasks.values():
        if t.status in (Status.running, Status.cancelling, Status.waiting):
            t.status = Status.cancelled

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

for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, handle_termination)

# 프로그램 시작 시 tasks 로드
load_tasks()

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def is_vllm_health():
    """Check if vllm server is reachable"""
    client = openai.AsyncOpenAI(base_url=base_url, api_key="dummy_key")
    try:
        await client.models.list()
        return True
    except Exception:
        return False

@app.get("/videos/{video_path:path}")
async def get_video(request: Request, video_path: str):
    video_path = os.path.join(UPLOAD_DIR, video_path)
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

@app.get("/browse/")
async def browse(path: str = ""):
    base_dir = os.path.abspath(UPLOAD_DIR)
    target = os.path.abspath(os.path.join(base_dir, path))
    if not target.startswith(base_dir) or not os.path.isdir(target):
        raise HTTPException(status_code=400, detail="Invalid path")
    entries = []
    for entry in os.scandir(target):
        entries.append({"name": entry.name, "is_dir": entry.is_dir()})
    rel_path = os.path.relpath(target, base_dir)
    if rel_path == ".":
        rel_path = ""
    return {"path": rel_path, "entries": entries}

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
    start_time: Optional[int] = Form(0),
    end_time: Optional[int] = Form(None)
):
    # 비디오 파일 존재 여부 및 재생 시간(초) 계산
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="비디오 파일을 찾을 수 없습니다.")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="비디오 파일을 열 수 없습니다.")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="FPS 값을 확인할 수 없습니다.")
    duration = frame_count / fps  # 비디오의 총 재생 시간(초)
    cap.release()
    
    # 범위 체크: start_time과 end_time이 올바른지 검증
    if start_time < 0:
        raise HTTPException(status_code=400, detail="시작 시간은 0보다 작을 수 없습니다.")
    if start_time >= duration:
        raise HTTPException(
            status_code=400, 
            detail=f"시작 시간({start_time}s)이 비디오 재생 시간({duration:.2f}s)보다 큽니다."
        )
    if end_time is not None:
        if end_time > duration:
            raise HTTPException(
                status_code=400, 
                detail=f"종료 시간({end_time}s)이 비디오 재생 시간({duration:.2f}s)보다 큽니다."
            )
        if end_time <= start_time:
            raise HTTPException(status_code=400, detail="종료 시간은 시작 시간보다 커야 합니다.")
    
    # 작업(task) 생성
    task_id = str(uuid.uuid4())
    tasks[task_id] = Task(
        task_id=task_id,
        video_filename=video_filename,
        status=Status.waiting,
        progress=0,
        estimated_completion="TBD",
        messages=[],
        task_start_time=None,
        ocr_x=x,
        ocr_y=y,
        ocr_width=width,
        ocr_height=height,
        ocr_start_time=start_time,
        ocr_end_time=end_time,
    )

    # 생성된 작업을 즉시 브로드캐스트하여 클라이언트에 표시
    await broadcast_update(tasks[task_id])

    # 대기 중인 작업이 하나뿐이고 실행 중인 작업이 없다면 바로 실행
    await start_next_task()
    
    return {"task_id": task_id}

async def run_ocr_task(task_id, video_filename, x, y, width, height, start_time, end_time):
    print(f"[시작] {task_id} - {video_filename}")

    task = tasks[task_id]

    try:
        # Ensure the vLLM container is running; start it if needed.
        if not await is_vllm_health():
            print("vLLM 서버가 준비되지 않았습니다. 컨테이너를 시작합니다.")
            try:
                docker_manager.start_container(docker_name)
            except (APIError, DockerException) as e:
                error_detail = getattr(e, "explanation", None) or str(e)
                task.status = Status.failed
                task.error = f"Docker 컨테이너 시작 실패: {error_detail}"
                await broadcast_update(task)
                print(f"Docker 컨테이너 시작 실패: {error_detail}")
                return
            except Exception as e:
                task.status = Status.failed
                task.error = f"컨테이너 시작 중 알 수 없는 오류: {e}"
                await broadcast_update(task)
                print("컨테이너 시작 중 알 수 없는 오류:", e)
                return

        # Wait until the vLLM server becomes healthy.
        while not await is_vllm_health():
            await asyncio.sleep(5)
        print("vLLM 서버가 준비되었습니다.")

        task.status = Status.running
        task.task_start_time = None
        await broadcast_update(task)
        async for progress in process_ocr(video_filename, x, y, width, height, start_time, end_time):
            # Check if cancellation was requested.
            if task.status == Status.cancelling:
                task.status = Status.cancelled
                await broadcast_update(task)
                return  # 작업 중단

            try:
                task.progress = progress
                task.estimated_completion = calculate_estimated_completion(task_id)
                await broadcast_update(task)
            except Exception as e:
                print("Progress message 처리 오류:", e)

        # Handle successful OCR completion.
        task.status = Status.completed
        filename_without_ext = os.path.splitext(os.path.basename(video_filename))[0]
        srt_path = os.path.join(UPLOAD_DIR, f"{filename_without_ext}.srt")
        task.result = srt_path if os.path.exists(srt_path) else None
        producer.send("discord_bot", {
            "type": "msg",
            "msg": f"{video_filename} OCR 완료."
        })
        await broadcast_update(task)
    except Exception as e:
        task.status = Status.failed
        task.error = str(e)
        print("OCR 작업 오류:", e)
        traceback.print_exc()
        await broadcast_update(task)
    finally:
        # Kick off the next task after finishing the current one.
        await start_next_task()


async def start_next_task():
    """대기 상태인 다음 작업이 있으면 실행합니다."""

    # 현재 실행 중인 작업이 있다면 다음 작업을 시작하지 않습니다.
    if get_running_task_id() is not None:
        return
    next_id = get_next_waiting_task_id()
    if not next_id:
        # 대기 중인 작업이 없으면 vLLM 컨테이너 중지로 자원 절약
        docker_manager.stop_container(docker_name)
        return

    next_task = tasks[next_id]
    asyncio.create_task(
        run_ocr_task(
            next_id,
            next_task.video_filename,
            next_task.ocr_x,
            next_task.ocr_y,
            next_task.ocr_width,
            next_task.ocr_height,
            next_task.ocr_start_time,
            next_task.ocr_end_time,
        )
    )


@app.post("/cancel_ocr/")
async def cancel_ocr(task_id: str = Form(...)):
    if task_id in tasks:
        task = tasks[task_id]
        if task.status == Status.waiting:
            task.status = Status.cancelled
            await broadcast_update(task)
            await start_next_task()
            return {"detail": f"Task {task_id} 취소되었습니다."}
        task.status = Status.cancelling
        await broadcast_update(task)
        return {"detail": f"Task {task_id} 취소 요청이 접수되었습니다."}
    else:
        raise HTTPException(status_code=404, detail="Task not found")


@app.post("/resume_ocr/")
async def resume_ocr(task_id: str = Form(...)):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task.status != Status.cancelled:
        raise HTTPException(status_code=400, detail="Task is not cancelled")
    task.status = Status.waiting
    task.task_start_time = None
    await broadcast_update(task)
    await start_next_task()
    return {"detail": f"Task {task_id} resumed"}
    

async def broadcast_update(task: Task):
    """
    모든 연결된 클라이언트 WebSocket에 message(JSON)를 전송합니다.
    전송에 실패한 경우 해당 WebSocket은 리스트에서 제거합니다.
    """
    message = asdict(task)

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
    return {tid: asdict(task) for tid, task in tasks.items()}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return asdict(tasks[task_id])


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
        video_filename = tasks[task_id].video_filename
        if video_filename:
            video_path = os.path.join(UPLOAD_DIR, video_filename)
            video_path_obj = Path(video_path)
            csv_path = str(video_path_obj.with_suffix(".csv"))
            if os.path.exists(csv_path):
                os.remove(csv_path)
        del tasks[task_id]


@app.delete("/tasks/{task_id}")
async def delete_task_endpoint(task_id: str):
    delete_task(task_id)
    return {"detail": f"Task {task_id} 삭제 요청이 처리되었습니다."}


# ---------------------------
# 예상 완료 시간 계산 함수 구현
# ---------------------------
def calculate_estimated_completion(task_id: int) -> str:
    """
    작업 정보를 바탕으로 작업 완료까지 남은 시간을 계산하여 HH:mm:ss 단위로 반환합니다.
    진행률이 0이면 TBD 문자열을 반환합니다.
    """

    task = tasks.get(task_id)
    progress = task.progress if task else 0
    task_start_time = task.task_start_time if task else None

    # 첫 프레임 처리 시작 시간 기록
    if task_start_time is None:
        task_start_time = time.time()
        if task:
            task.task_start_time = task_start_time

    if not task_start_time or progress <= 0:
        return "TBD"

    elif progress == 100:
        return "00:00:00"
    elapsed = time.time() - task_start_time
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

