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
from dataclasses import dataclass, asdict
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
from pydantic import BaseModel, ValidationError, ConfigDict

from core.ocr import process_ocr, UPLOAD_DIR, OcrProcessingError, is_detector_cache_complete
from core.docker_manager import DockerManager
from core.settings_manager import AppSettings, settings_manager

class Status(str, Enum):
    waiting = "waiting"
    running = "running"
    completed = "completed"
    fatal_error = "fatal_error"
    retryable_error = "retryable_error"
    stopping = "stopping"
    stopped = "stopped"

@dataclass
class Task:
    task_id: str
    video_filename: str
    status: Status
    progress: int = 0
    estimated_completion: str = "TBD"
    task_start_time: Optional[float] = None
    ocr_x: int = 0
    ocr_y: int = 0
    ocr_width: int = 0
    ocr_height: int = 0
    ocr_start_time: Optional[int] = 0
    ocr_end_time: Optional[int] = None
    full_screen_ocr: bool = True
    mask_x: Optional[int] = None
    mask_y: Optional[int] = None
    mask_width: Optional[int] = None
    mask_height: Optional[int] = None
    result: Optional[str] = None
    error: Optional[str] = None


class SettingsUpdateRequest(BaseModel):
    docker_enabled: Optional[bool] = None
    docker_url: Optional[str] = None
    detector_docker_name: Optional[str] = None
    recognizer_docker_name: Optional[str] = None
    kafka_enabled: Optional[bool] = None
    kafka_url: Optional[str] = None
    detector_llm_base_url: Optional[str] = None
    detector_llm_model: Optional[str] = None
    recognizer_llm_base_url: Optional[str] = None
    recognizer_llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

current_settings: AppSettings = settings_manager.get_settings()
docker_manager: Optional[DockerManager] = None
kafka_producer: Optional[KafkaProducer] = None

DETECTOR_ROLE = "detector"
RECOGNIZER_ROLE = "recognizer"


def create_kafka_producer(settings: AppSettings) -> Optional[KafkaProducer]:
    if not settings.kafka_enabled:
        return None
    try:
        return KafkaProducer(
            acks=0,
            compression_type="gzip",
            bootstrap_servers=[settings.kafka_url],
            value_serializer=lambda x: dumps(x).encode("utf-8"),
        )
    except Exception as exc:
        print("KafkaProducer 초기화 실패:", exc)
        return None


def apply_runtime_settings(settings: AppSettings) -> None:
    global docker_manager, kafka_producer, current_settings

    current_settings = settings
    docker_manager = None
    if settings.docker_enabled:
        try:
            docker_manager = DockerManager(settings.docker_url)
        except Exception as exc:
            print("DockerManager 초기화 실패:", exc)
            docker_manager = None

    if kafka_producer is not None:
        try:
            kafka_producer.close()
        except Exception:
            pass

    kafka_producer = create_kafka_producer(settings)


def get_role_vllm_config(role: str, settings: AppSettings | None = None) -> dict[str, str | None]:
    target_settings = settings or current_settings
    if role == DETECTOR_ROLE:
        return {
            "role_name": "BBox Detector",
            "docker_name": target_settings.detector_docker_name,
            "base_url": target_settings.detector_llm_base_url,
            "model": target_settings.detector_llm_model,
        }
    if role == RECOGNIZER_ROLE:
        return {
            "role_name": "OCR Recognizer",
            "docker_name": target_settings.recognizer_docker_name,
            "base_url": target_settings.recognizer_llm_base_url,
            "model": target_settings.recognizer_llm_model,
        }
    raise ValueError(f"알 수 없는 vLLM 역할입니다: {role}")


def get_opposite_role(role: str) -> str:
    if role == DETECTOR_ROLE:
        return RECOGNIZER_ROLE
    if role == RECOGNIZER_ROLE:
        return DETECTOR_ROLE
    raise ValueError(f"알 수 없는 vLLM 역할입니다: {role}")


def iter_configured_container_names(settings: AppSettings | None = None) -> list[str]:
    target_settings = settings or current_settings
    names = [
        target_settings.detector_docker_name,
        target_settings.recognizer_docker_name,
    ]
    unique_names: list[str] = []
    for name in names:
        cleaned_name = (name or "").strip()
        if cleaned_name and cleaned_name not in unique_names:
            unique_names.append(cleaned_name)
    return unique_names


def publish_kafka_message(topic: str, payload: dict) -> None:
    if kafka_producer is None:
        return
    try:
        kafka_producer.send(topic, payload)
    except Exception as exc:
        print("Kafka 메시지 전송 실패:", exc)


def shutdown_runtime_services() -> None:
    if kafka_producer is None:
        return
    try:
        kafka_producer.flush()
        kafka_producer.close()
    except Exception:
        pass


apply_runtime_settings(current_settings)

app = FastAPI()

# 업로드된 비디오 파일 저장 경로
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# 글로벌 작업 상태 저장소
# 예: { task_id: { "video_filename": str, "status": Status,
#                   "progress": 0~100, "messages": [progress update objects],
#                   "result": srt 파일 경로, "error": str, "task_start_time": timestamp } }
PICKLE_FILENAME = 'tasks.pkl'
tasks: Dict[str, Task] = {}


def get_active_task_id_by_video_filename(video_filename: str) -> Optional[str]:
    """같은 비디오 경로로 실행/대기 중인 작업 ID를 반환합니다."""
    target_video_path = os.path.abspath(os.path.join(UPLOAD_DIR, video_filename))
    active_statuses = {Status.waiting, Status.running, Status.stopping}
    for task_id, task in tasks.items():
        if task.status not in active_statuses:
            continue
        task_video_path = os.path.abspath(os.path.join(UPLOAD_DIR, task.video_filename))
        if task_video_path == target_video_path:
            return task_id
    return None


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
                            data.pop('messages', None)
                            data.pop('subtitle_color_enabled', None)
                            data.pop('subtitle_color_ranges', None)
                            data['status'] = Status(data.get('status', Status.fatal_error))
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
    
    # 로드된 테스크 정보에서 실행 중이던 상태를 모두 stopped 로 변경
    for t in tasks.values():
        if t.status in (Status.running, Status.stopping, Status.waiting):
            t.status = Status.stopped

def save_tasks():
    try:
        with open(PICKLE_FILENAME, 'wb') as f:
            pickle.dump(tasks, f)
        print(f"tasks를 {PICKLE_FILENAME}에 저장했습니다.")
    except Exception as e:
        print("tasks 저장 중 오류 발생:", e)

# atexit를 사용해 프로그램 종료 시 자동 저장 및 리소스 정리
atexit.register(shutdown_runtime_services)
atexit.register(save_tasks)

def handle_termination(signum, frame):
    print(f"종료 시그널({signum}) 수신 - tasks 저장 중...")
    save_tasks()
    shutdown_runtime_services()
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
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/settings", response_class=HTMLResponse)
async def read_settings_page(request: Request):
    return templates.TemplateResponse(request=request, name="settings.html")


@app.get("/api/settings")
async def get_settings_api():
    settings = settings_manager.get_settings()
    return settings.model_dump()


@app.get("/api/docker/containers")
async def get_docker_containers_api(docker_url: Optional[str] = None):
    target_docker_url = (docker_url or current_settings.docker_url or "").strip()
    if not target_docker_url:
        raise HTTPException(status_code=400, detail="Docker 엔드포인트를 입력해주세요.")

    try:
        target_docker_manager = DockerManager(target_docker_url)
        containers = target_docker_manager.list_containers()
    except (APIError, DockerException) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Docker 컨테이너 목록을 불러오지 못했습니다",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Docker 연결 중 오류가 발생했습니다: {exc}") from exc

    return {"docker_url": target_docker_url, "containers": containers}


@app.put("/api/settings")
async def update_settings_api(payload: SettingsUpdateRequest):
    updates = payload.model_dump(exclude_unset=True)
    docker_enabled = updates.get("docker_enabled")
    if docker_enabled is None:
        docker_enabled = current_settings.docker_enabled

    if docker_enabled:
        docker_url_value = updates.get("docker_url", current_settings.docker_url)
        detector_docker_name_value = updates.get("detector_docker_name", current_settings.detector_docker_name)
        recognizer_docker_name_value = updates.get("recognizer_docker_name", current_settings.recognizer_docker_name)
        if not docker_url_value or not docker_url_value.strip():
            raise HTTPException(status_code=400, detail="Docker 엔드포인트를 입력해주세요.")
        if not detector_docker_name_value or not detector_docker_name_value.strip():
            raise HTTPException(status_code=400, detail="BBox Detector 컨테이너 이름을 선택해주세요.")
        if not recognizer_docker_name_value or not recognizer_docker_name_value.strip():
            raise HTTPException(status_code=400, detail="OCR Recognizer 컨테이너 이름을 선택해주세요.")
        if detector_docker_name_value.strip() == recognizer_docker_name_value.strip():
            raise HTTPException(status_code=400, detail="Detector와 Recognizer는 서로 다른 컨테이너를 선택해주세요.")
        try:
            target_docker_manager = DockerManager(docker_url_value.strip())
            container_names = target_docker_manager.list_containers()
        except (APIError, DockerException) as exc:
            raise HTTPException(status_code=400, detail="Docker 연결을 확인할 수 없습니다.") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Docker 연결 중 오류가 발생했습니다: {exc}") from exc
        container_name_set = set(container_names)
        for role_label, docker_name_value in (
            ("BBox Detector", detector_docker_name_value),
            ("OCR Recognizer", recognizer_docker_name_value),
        ):
            if docker_name_value.strip() not in container_name_set:
                raise HTTPException(
                    status_code=400,
                    detail=f"{role_label} 컨테이너를 찾을 수 없습니다: {docker_name_value.strip()}",
                )

    kafka_enabled = updates.get("kafka_enabled")
    if kafka_enabled is None:
        kafka_enabled = current_settings.kafka_enabled
    if kafka_enabled:
        kafka_url_value = updates.get("kafka_url", current_settings.kafka_url)
        if not kafka_url_value or not kafka_url_value.strip():
            raise HTTPException(status_code=400, detail="Kafka URL을 입력해주세요.")
    preview_settings = AppSettings(
        **{
            **current_settings.model_dump(),
            **updates,
        }
    ).normalized()
    if not preview_settings.detector_llm_base_url:
        raise HTTPException(status_code=400, detail="BBox Detector LLM Base URL을 입력해주세요.")
    if not preview_settings.detector_llm_model:
        raise HTTPException(status_code=400, detail="BBox Detector 모델명을 입력해주세요.")
    if not preview_settings.recognizer_llm_base_url:
        raise HTTPException(status_code=400, detail="OCR Recognizer LLM Base URL을 입력해주세요.")
    if not preview_settings.recognizer_llm_model:
        raise HTTPException(status_code=400, detail="OCR Recognizer 모델명을 입력해주세요.")
    try:
        new_settings = settings_manager.update(**updates)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors())
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"설정을 저장할 수 없습니다: {exc}") from exc

    apply_runtime_settings(new_settings)
    return new_settings.model_dump()

async def is_vllm_health(role: str):
    """Check if vllm server is reachable"""
    config = get_role_vllm_config(role)
    base_url = (config["base_url"] or "").strip()
    if not base_url:
        return False
    client = openai.AsyncOpenAI(
        base_url=base_url,
        api_key=current_settings.llm_api_key or "dummy_key",
    )
    try:
        models = await client.models.list()
        model_ids = [item.id for item in getattr(models, "data", []) if getattr(item, "id", None)]
        if model_ids:
            print(f"{config['role_name']} vLLM 모델 목록: {', '.join(model_ids)}")
        return True
    except Exception as exc:
        print(f"{config['role_name']} vLLM 헬스체크 실패: {exc}")
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
    entries.sort(key=lambda item: (not item["is_dir"], item["name"].casefold()))
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
    end_time: Optional[int] = Form(None),
    full_screen_ocr: bool = Form(True),
    mask_x: Optional[int] = Form(None),
    mask_y: Optional[int] = Form(None),
    mask_width: Optional[int] = Form(None),
    mask_height: Optional[int] = Form(None),
):
    if not current_settings.docker_enabled:
        raise HTTPException(status_code=400, detail="두 vLLM 컨테이너의 순차 실행을 보장하려면 Docker 자동 제어를 활성화해야 합니다.")
    if full_screen_ocr and not current_settings.detector_llm_base_url:
        raise HTTPException(status_code=400, detail="BBox Detector LLM Base URL 설정이 필요합니다.")
    if full_screen_ocr and not current_settings.detector_llm_model:
        raise HTTPException(status_code=400, detail="BBox Detector 모델 설정이 필요합니다.")
    if not current_settings.recognizer_llm_base_url:
        raise HTTPException(status_code=400, detail="OCR Recognizer LLM Base URL 설정이 필요합니다.")
    if not current_settings.recognizer_llm_model:
        raise HTTPException(status_code=400, detail="OCR Recognizer 모델 설정이 필요합니다.")

    duplicate_task_id = get_active_task_id_by_video_filename(video_filename)
    if duplicate_task_id is not None:
        raise HTTPException(
            status_code=409,
            detail=f"동일한 비디오 파일 작업이 이미 진행 중입니다. (task_id: {duplicate_task_id})",
        )

    # 비디오 파일 존재 여부 및 재생 시간(초) 계산
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="비디오 파일을 찾을 수 없습니다.")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="비디오 파일을 열 수 없습니다.")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="FPS 값을 확인할 수 없습니다.")
    duration = frame_count / fps  # 비디오의 총 재생 시간(초)
    cap.release()

    if full_screen_ocr:
        x = 0
        y = 0
        width = frame_width
        height = frame_height
    else:
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="OCR width and height must be greater than 0.")
        if x < 0 or y < 0:
            raise HTTPException(status_code=400, detail="OCR x and y must be greater than or equal to 0.")
        if x + width > frame_width or y + height > frame_height:
            raise HTTPException(status_code=400, detail="OCR crop area must stay within the video frame.")
    
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
    has_any_mask_value = any(value is not None for value in (mask_x, mask_y, mask_width, mask_height))
    has_all_mask_values = all(value is not None for value in (mask_x, mask_y, mask_width, mask_height))
    if has_any_mask_value and not has_all_mask_values:
        raise HTTPException(status_code=400, detail="Mask coordinates must include x, y, width, and height.")

    has_mask = has_all_mask_values
    if has_mask and full_screen_ocr:
        raise HTTPException(status_code=400, detail="Mask is only supported in crop OCR mode.")

    if has_mask:
        if mask_width <= 0 or mask_height <= 0:
            raise HTTPException(status_code=400, detail="Mask width and height must be greater than 0.")
        if mask_x < 0 or mask_y < 0:
            raise HTTPException(status_code=400, detail="Mask x and y must be greater than or equal to 0.")
        if mask_x + mask_width > frame_width or mask_y + mask_height > frame_height:
            raise HTTPException(status_code=400, detail="Mask area must stay within the video frame.")
        if (
            mask_x < x or
            mask_y < y or
            mask_x + mask_width > x + width or
            mask_y + mask_height > y + height
        ):
            raise HTTPException(status_code=400, detail="Mask area must stay within the OCR crop area.")

    task_id = str(uuid.uuid4())
    tasks[task_id] = Task(
        task_id=task_id,
        video_filename=video_filename,
        status=Status.waiting,
        progress=0,
        estimated_completion="TBD",
        task_start_time=None,
        ocr_x=x,
        ocr_y=y,
        ocr_width=width,
        ocr_height=height,
        ocr_start_time=start_time,
        ocr_end_time=end_time,
        full_screen_ocr=full_screen_ocr,
        mask_x=mask_x if has_mask else None,
        mask_y=mask_y if has_mask else None,
        mask_width=mask_width if has_mask else None,
        mask_height=mask_height if has_mask else None,
    )

    # 생성된 작업을 즉시 브로드캐스트하여 클라이언트에 표시
    await broadcast_update(tasks[task_id])

    # 대기 중인 작업이 하나뿐이고 실행 중인 작업이 없다면 바로 실행
    await start_next_task()
    
    return {"task_id": task_id}


async def fail_task(task: Task, message: str) -> None:
    task.status = Status.fatal_error
    task.error = message
    await broadcast_update(task)


async def ensure_vllm_role(role: str, task: Task) -> bool:
    config = get_role_vllm_config(role)
    role_name = str(config["role_name"])
    target_container_name = (config["docker_name"] or "").strip()

    if not current_settings.docker_enabled:
        await fail_task(task, "두 vLLM 컨테이너의 순차 실행을 보장하려면 Docker 자동 제어를 활성화해야 합니다.")
        return False
    if docker_manager is None:
        await fail_task(task, "Docker 자동 제어가 활성화되어 있지만 DockerManager를 초기화하지 못했습니다.")
        return False
    if not target_container_name:
        await fail_task(task, f"{role_name} 컨테이너 이름 설정이 필요합니다.")
        return False

    opposite_config = get_role_vllm_config(get_opposite_role(role))
    opposite_container_name = (opposite_config["docker_name"] or "").strip()
    if opposite_container_name and opposite_container_name != target_container_name:
        try:
            docker_manager.stop_container(opposite_container_name)
        except (APIError, DockerException) as exc:
            error_detail = getattr(exc, "explanation", None) or str(exc)
            await fail_task(task, f"{opposite_config['role_name']} 컨테이너 중지 실패: {error_detail}")
            return False
        except Exception as exc:
            await fail_task(task, f"{opposite_config['role_name']} 컨테이너 중지 중 알 수 없는 오류: {exc}")
            return False

    if await is_vllm_health(role):
        print(f"{role_name} vLLM 서버가 준비되었습니다.")
        return True

    print(f"{role_name} vLLM 서버가 준비되지 않았습니다. 컨테이너를 시작합니다.")
    try:
        docker_manager.start_container(target_container_name)
    except (APIError, DockerException) as exc:
        error_detail = getattr(exc, "explanation", None) or str(exc)
        await fail_task(task, f"{role_name} Docker 컨테이너 시작 실패: {error_detail}")
        print(f"{role_name} Docker 컨테이너 시작 실패: {error_detail}")
        return False
    except Exception as exc:
        await fail_task(task, f"{role_name} 컨테이너 시작 중 알 수 없는 오류: {exc}")
        print(f"{role_name} 컨테이너 시작 중 알 수 없는 오류:", exc)
        return False

    for _ in range(60):
        if await is_vllm_health(role):
            print(f"{role_name} vLLM 서버가 준비되었습니다.")
            return True
        await asyncio.sleep(5)
    await fail_task(task, f"{role_name} vLLM 서버가 제한 시간 안에 준비되지 않았습니다.")
    return False


async def run_ocr_task(
    task_id,
    video_filename,
    x,
    y,
    width,
    height,
    start_time,
    end_time,
    full_screen_ocr=True,
    mask_x=None,
    mask_y=None,
    mask_width=None,
    mask_height=None,
):
    print(f"[시작] {task_id} - {video_filename}")

    task = tasks[task_id]

    try:
        detector_cache_complete = (
            True
            if not full_screen_ocr
            else is_detector_cache_complete(video_filename, start_time, end_time)
        )
        if full_screen_ocr and not detector_cache_complete and not current_settings.detector_llm_base_url:
            raise RuntimeError("BBox Detector LLM Base URL 설정이 필요합니다.")
        if full_screen_ocr and not detector_cache_complete and not current_settings.detector_llm_model:
            raise RuntimeError("BBox Detector 모델 설정이 필요합니다.")
        if not current_settings.recognizer_llm_base_url:
            raise RuntimeError("OCR Recognizer LLM Base URL 설정이 필요합니다.")
        if not current_settings.recognizer_llm_model:
            raise RuntimeError("OCR Recognizer 모델 설정이 필요합니다.")

        initial_role = RECOGNIZER_ROLE if detector_cache_complete else DETECTOR_ROLE
        if not await ensure_vllm_role(initial_role, task):
            return

        task.status = Status.running
        task.progress = 1
        task.task_start_time = None
        await broadcast_update(task)

        async def switch_to_recognizer() -> bool:
            if task.status == Status.stopping:
                return False
            if not await ensure_vllm_role(RECOGNIZER_ROLE, task):
                raise RuntimeError(task.error or "OCR Recognizer vLLM 서버를 준비하지 못했습니다.")
            return True

        async for progress in process_ocr(
            video_filename,
            x,
            y,
            width,
            height,
            start_time,
            end_time,
            full_screen_ocr=full_screen_ocr,
            mask_x=mask_x,
            mask_y=mask_y,
            mask_width=mask_width,
            mask_height=mask_height,
            switch_to_recognizer=switch_to_recognizer,
        ):
            # 중지 요청이 들어오면 현재 진행 중인 OCR 을 중단합니다.
            if task.status == Status.stopping:
                task.status = Status.stopped
                await broadcast_update(task)
                return  # 작업 중단

            try:
                task.progress = progress
                task.estimated_completion = calculate_estimated_completion(task_id)
                await broadcast_update(task)
            except Exception as e:
                print("Progress message 처리 오류:", e)

        if task.status == Status.stopping:
            task.status = Status.stopped
            await broadcast_update(task)
            return

        # OCR 완료 상태를 처리합니다.
        task.status = Status.completed
        filename_without_ext = os.path.splitext(os.path.basename(video_filename))[0]
        srt_path = os.path.join(UPLOAD_DIR, f"{filename_without_ext}.srt")
        task.result = srt_path if os.path.exists(srt_path) else None
        publish_kafka_message("discord_bot", {
            "type": "msg",
            "msg": f"{video_filename} OCR 완료."
        })
        await broadcast_update(task)
    except OcrProcessingError as e:
        task.status = Status.retryable_error
        task.error = str(e)
        print("OCR 프레임 처리 중 오류:", e)
        traceback.print_exc()
        await broadcast_update(task)
    except Exception as e:
        task.status = Status.fatal_error
        task.error = str(e)
        print("OCR 작업 오류:", e)
        traceback.print_exc()
        await broadcast_update(task)
    finally:
        # 현재 작업이 끝나면 다음 작업을 시작합니다.
        await start_next_task()


async def start_next_task():
    """대기 상태인 다음 작업이 있으면 실행합니다."""

    # 현재 실행 중인 작업이 있다면 다음 작업을 시작하지 않습니다.
    if get_running_task_id() is not None:
        return
    next_id = get_next_waiting_task_id()
    if not next_id:
        if current_settings.docker_enabled and docker_manager is not None:
            for container_name in iter_configured_container_names():
                try:
                    docker_manager.stop_container(container_name)
                except Exception as exc:
                    print("Docker 컨테이너 중지 실패:", exc)
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
            next_task.full_screen_ocr,
            next_task.mask_x,
            next_task.mask_y,
            next_task.mask_width,
            next_task.mask_height,
        )
    )


@app.post("/stop_ocr/")
async def stop_ocr(task_id: str = Form(...)):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    if task.status == Status.stopping:
        return {"detail": f"Task {task_id} 는 이미 중지 요청이 접수된 상태입니다."}
    if task.status == Status.stopped:
        return {"detail": f"Task {task_id} 는 이미 중지된 상태입니다."}
    if task.status not in (Status.waiting, Status.running):
        return {"detail": f"Task {task_id} 는 이 상태에서 중지될 수 없습니다. '{task.status.value}'"}

    if task.status == Status.waiting:
        task.status = Status.stopped
        await broadcast_update(task)
        await start_next_task()
        return {"detail": f"Task {task_id} 중지되었습니다."}
    else:
        task.status = Status.stopping
        await broadcast_update(task)
        return {"detail": f"Task {task_id} 중지 요청이 접수되었습니다."}


@app.post("/stop_all_ocr/")
async def stop_all_ocr():
    stopped_task_ids = []

    for task in tasks.values():
        if task.status == Status.waiting:
            task.status = Status.stopped
            stopped_task_ids.append(task.task_id)
            await broadcast_update(task)
        elif task.status == Status.running:
            task.status = Status.stopping
            stopped_task_ids.append(task.task_id)
            await broadcast_update(task)

    return {"detail": f"{len(stopped_task_ids)}개 작업의 중지 요청이 접수되었습니다."}


@app.post("/resume_ocr/")
async def resume_ocr(task_id: str = Form(...)):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task.status not in (Status.retryable_error, Status.stopped):
        raise HTTPException(status_code=400, detail="Task is not resumable")
    task.status = Status.waiting
    task.task_start_time = None
    task.error = None
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
    - 관련 파일(JSONL, SRT)이 존재하면 삭제
    """
    if task_id in tasks:
        video_filename = tasks[task_id].video_filename
        if video_filename:
            video_path = os.path.join(UPLOAD_DIR, video_filename)
            video_path_obj = Path(video_path)
            jsonl_path = str(video_path_obj.with_suffix(".jsonl"))
            if os.path.exists(jsonl_path):
                os.remove(jsonl_path)
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
