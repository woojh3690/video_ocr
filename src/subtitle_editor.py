import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.jsonl_to_srt import (
    DEFAULT_MERGE_PARAMS,
    MergeParams,
    Segment,
    build_preview_segments,
    write_srt,
)
from core.ocr import UPLOAD_DIR


SaveTasksFunc = Callable[[], None]
BroadcastFunc = Callable[[Any], Awaitable[None] | None]

templates = Jinja2Templates(directory="templates")


class MergeParamsPayload(BaseModel):
    # GUI 슬라이더와 같은 범위로 API 입력을 제한합니다.
    duplicate_gap_sec: float = Field(DEFAULT_MERGE_PARAMS.duplicate_gap_sec, ge=0.0, le=5.0)
    contained_gap_sec: float = Field(DEFAULT_MERGE_PARAMS.contained_gap_sec, ge=0.0, le=3.0)
    min_contained_key_len: int = Field(DEFAULT_MERGE_PARAMS.min_contained_key_len, ge=2, le=12)
    similar_threshold: float = Field(DEFAULT_MERGE_PARAMS.similar_threshold, ge=0.88, le=1.0)
    min_similar_key_len: int = Field(DEFAULT_MERGE_PARAMS.min_similar_key_len, ge=4, le=24)
    similar_length_ratio: float = Field(DEFAULT_MERGE_PARAMS.similar_length_ratio, ge=0.5, le=1.0)
    min_duration_sec: float = Field(DEFAULT_MERGE_PARAMS.min_duration_sec, ge=0.0, le=2.0)
    postprocess_passes: int = Field(DEFAULT_MERGE_PARAMS.postprocess_passes, ge=1, le=5)

    model_config = ConfigDict(extra="forbid")

    def to_merge_params(self) -> MergeParams:
        # core 계층에는 FastAPI/Pydantic 타입을 넘기지 않습니다.
        return MergeParams.from_mapping(self.model_dump())


def _status_value(task: Any) -> str:
    # Enum 또는 문자열 상태를 같은 방식으로 비교합니다.
    status = getattr(task, "status", "")
    return getattr(status, "value", status)


def _video_path_for_task(task: Any) -> Path:
    # OCR 저장 규칙과 동일하게 업로드 폴더 아래의 영상 파일을 찾습니다.
    return Path(UPLOAD_DIR) / getattr(task, "video_filename", "")


def _jsonl_path_for_task(task: Any) -> Path:
    # 영상 파일명과 같은 stem을 가진 JSONL이 OCR 결과입니다.
    return _video_path_for_task(task).with_suffix(".jsonl")


def _srt_path_for_task(task: Any, jsonl_path: Path) -> Path | None:
    # 기존 result를 우선 사용하고, 없으면 같은 stem의 기존 SRT를 찾아 저장 경로로 사용합니다.
    result = getattr(task, "result", None)
    if result:
        result_path = Path(result)
        if result_path.exists():
            return result_path

    candidates = list(jsonl_path.parent.glob(f"{jsonl_path.stem}.*.srt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _validate_completed_task(tasks: dict[str, Any], task_id: str) -> tuple[Any, Path, Path]:
    # 완료된 작업, 영상 파일, JSONL 파일이 모두 있어야 편집 페이지에 진입할 수 있습니다.
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if _status_value(task) != "completed":
        raise HTTPException(status_code=404, detail="Completed task not found")

    video_path = _video_path_for_task(task)
    jsonl_path = _jsonl_path_for_task(task)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    if not jsonl_path.exists():
        raise HTTPException(status_code=404, detail="JSONL file not found")
    return task, video_path, jsonl_path


def _parse_merge_params(payload: Any) -> MergeParamsPayload:
    # FastAPI 기본 422 대신 이 API 계약에 맞춰 400으로 파라미터 오류를 반환합니다.
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")

    raw_params = payload.get("params", {})
    if not isinstance(raw_params, dict):
        raise HTTPException(status_code=400, detail="params must be an object")

    try:
        return MergeParamsPayload(**raw_params)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc


def _segment_to_dict(segment: Segment) -> dict[str, Any]:
    # 프런트엔드에서 바로 사용할 수 있는 단순 JSON 구조로 변환합니다.
    return {
        "index": segment.index,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
    }


def _build_metrics(segments: list[Segment]) -> dict[str, Any]:
    # 미리보기 품질을 빠르게 확인할 수 있는 최소 메트릭을 제공합니다.
    durations = [max(0.0, segment.end - segment.start) for segment in segments]
    total_duration = sum(durations)
    return {
        "segment_count": len(segments),
        "total_duration_sec": round(total_duration, 3),
        "average_duration_sec": round(total_duration / len(segments), 3) if segments else 0.0,
        "min_duration_sec": round(min(durations), 3) if durations else 0.0,
        "max_duration_sec": round(max(durations), 3) if durations else 0.0,
        "multiline_count": sum(1 for segment in segments if "\n" in segment.text),
    }


def _video_url(video_path: Path) -> str:
    # /videos 라우트는 uploads 하위 상대 경로를 받으므로 URL 경로 세그먼트만 인코딩합니다.
    relative_path = video_path.relative_to(Path(UPLOAD_DIR)).as_posix()
    return f"/videos/{quote(relative_path, safe='/')}"


async def _broadcast_task_update(broadcast_update: BroadcastFunc, task: Any) -> None:
    # main.py의 websocket broadcast가 async이므로 호출 결과를 확인해 기다립니다.
    result = broadcast_update(task)
    if inspect.isawaitable(result):
        await result


def create_subtitle_editor_router(
    tasks: dict[str, Any],
    save_tasks: SaveTasksFunc,
    broadcast_update: BroadcastFunc,
) -> APIRouter:
    # main.py는 전역 상태만 주입하고, 실제 편집 기능은 이 router가 담당합니다.
    router = APIRouter()

    @router.get("/subtitle-editor/{task_id}", response_class=HTMLResponse)
    async def subtitle_editor_page(request: Request, task_id: str):
        _validate_completed_task(tasks, task_id)
        return templates.TemplateResponse(
            request=request,
            name="subtitle_editor.html",
            context={"task_id": task_id},
        )

    @router.get("/api/subtitle-editor/{task_id}")
    async def subtitle_editor_metadata(task_id: str):
        task, video_path, jsonl_path = _validate_completed_task(tasks, task_id)
        srt_path = _srt_path_for_task(task, jsonl_path)
        return {
            "task_id": task_id,
            "video_filename": getattr(task, "video_filename", ""),
            "video_url": _video_url(video_path),
            "srt_filename": srt_path.name if srt_path else None,
            "default_params": DEFAULT_MERGE_PARAMS.to_dict(),
            "current_params": DEFAULT_MERGE_PARAMS.to_dict(),
            "can_merge": True,
        }

    @router.post("/api/subtitle-editor/{task_id}/preview")
    async def subtitle_editor_preview(task_id: str, payload: Any = Body(default=None)):
        _task, _video_path, jsonl_path = _validate_completed_task(tasks, task_id)
        params_payload = _parse_merge_params(payload)
        params = params_payload.to_merge_params()
        segments = build_preview_segments(jsonl_path, params=params)
        return {
            "segments": [_segment_to_dict(segment) for segment in segments],
            "metrics": _build_metrics(segments),
            "params": params.to_dict(),
        }

    @router.post("/api/subtitle-editor/{task_id}/merge")
    async def subtitle_editor_merge(task_id: str, payload: Any = Body(default=None)):
        task, _video_path, jsonl_path = _validate_completed_task(tasks, task_id)
        params_payload = _parse_merge_params(payload)
        params = params_payload.to_merge_params()
        segments = build_preview_segments(jsonl_path, params=params)
        existing_srt_path = _srt_path_for_task(task, jsonl_path)
        srt_path = write_srt(jsonl_path, segments, target_path=existing_srt_path)
        task.result = str(srt_path)
        save_tasks()
        await _broadcast_task_update(broadcast_update, task)
        return {
            "segments": [_segment_to_dict(segment) for segment in segments],
            "metrics": _build_metrics(segments),
            "params": params.to_dict(),
            "srt_filename": srt_path.name,
        }

    return router
