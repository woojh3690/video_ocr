import asyncio
import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Generator, Iterable, List

import cv2

from core.jsonl_to_srt import jsonl_to_srt
from core.logging_utils import log_ocr_event
from core.ocr_types import SpottingItem
from core.settings_manager import get_settings
from core.split_ocr_client import (
    PaddleOCRRecognizerClient,
    SuryaDetectorClient,
    TextBlock,
    crop_with_padding,
    spotting_item_from_block,
)
from core.vllm_load_controller import AdaptiveConcurrencyController, VllmMetricsClient

UPLOAD_DIR = "uploads"


def read_positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


# 적응형 동시성 설정은 잘못된 환경 변수 값을 안전한 기본값으로 되돌립니다.
def read_non_negative_int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def read_positive_float_env(name: str, default: float) -> float:
    try:
        return max(0.1, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


DETECTOR_CONCURRENCY = read_positive_int_env("DETECTOR_CONCURRENCY", 16)
RECOGNIZER_CONCURRENCY = read_positive_int_env("RECOGNIZER_CONCURRENCY", 16)
DETECTOR_MIN_CONCURRENCY = read_positive_int_env("DETECTOR_MIN_CONCURRENCY", 1)
RECOGNIZER_MIN_CONCURRENCY = read_positive_int_env("RECOGNIZER_MIN_CONCURRENCY", 1)
DETECTOR_MAX_CONCURRENCY = read_positive_int_env("DETECTOR_MAX_CONCURRENCY", 64)
RECOGNIZER_MAX_CONCURRENCY = read_positive_int_env("RECOGNIZER_MAX_CONCURRENCY", 64)
VLLM_ADAPTIVE_CONCURRENCY = read_bool_env("VLLM_ADAPTIVE_CONCURRENCY", True)
VLLM_TARGET_WAITING = read_non_negative_int_env("VLLM_TARGET_WAITING", 2)
VLLM_METRICS_POLL_INTERVAL_SEC = read_positive_float_env("VLLM_METRICS_POLL_INTERVAL_SEC", 1.0)


@dataclass(frozen=True, slots=True)
class OcrPaths:
    video_path: Path
    jsonl_path: Path


@dataclass(slots=True)
class OcrJsonlState:
    detector_blocks_by_frame: dict[int, list[TextBlock]]
    ocr_records_by_frame: dict[int, dict[str, Any]]

    @property
    def ocr_frame_numbers(self) -> set[int]:
        return set(self.ocr_records_by_frame.keys())


@dataclass(slots=True)
class RecognizerFrameState:
    remaining: int
    items: list[SpottingItem | None]


def get_ocr_paths(video_filename: str) -> OcrPaths:
    video_path = Path(UPLOAD_DIR) / video_filename
    return OcrPaths(
        video_path=video_path,
        jsonl_path=video_path.with_suffix(".jsonl"),
    )


def get_required_frame_numbers(start_frame: int, end_frame: int) -> set[int]:
    return set(range(start_frame + 1, end_frame + 1))


def phase_progress(processed: int, total: int, start: float, end: float, cap: float) -> float:
    span = end - start
    value = start + (processed / max(1, total)) * span
    return round(min(cap, value), 2)


async def wait_for_completed_tasks(pending_tasks: set[asyncio.Task]):
    return await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)


async def collect_completed_progress(
    pending_tasks: set[asyncio.Task],
    flush_completed_tasks: Callable[[set[asyncio.Task]], Awaitable[list[float]]],
) -> tuple[set[asyncio.Task], list[float]]:
    done_tasks, pending_tasks = await wait_for_completed_tasks(pending_tasks)
    return pending_tasks, await flush_completed_tasks(done_tasks)


async def wait_for_submission_slot(
    pending_tasks: set[asyncio.Task],
    flush_completed_tasks: Callable[[set[asyncio.Task]], Awaitable[list[float]]],
    concurrency_controller: AdaptiveConcurrencyController,
) -> tuple[set[asyncio.Task], list[float]]:
    # 메트릭 I/O 없이 마지막 동시성 제한만 확인해 요청 제출이 지연되지 않게 합니다.
    progress_values: list[float] = []
    while pending_tasks and len(pending_tasks) >= concurrency_controller.limit:
        pending_tasks, completed_progress = await collect_completed_progress(
            pending_tasks,
            flush_completed_tasks,
        )
        progress_values.extend(completed_progress)
    return pending_tasks, progress_values


def parse_serialized_detector_blocks(blocks_data: Any) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    if not isinstance(blocks_data, list):
        return blocks

    for block in blocks_data:
        if not isinstance(block, dict):
            continue
        try:
            normalized_bbox = tuple(int(value) for value in block["normalized_bbox"])
            pixel_bbox = tuple(int(value) for value in block["pixel_bbox"])
        except (KeyError, TypeError, ValueError):
            continue
        if len(normalized_bbox) != 4 or len(pixel_bbox) != 4:
            continue
        blocks.append(
            TextBlock(
                normalized_bbox=normalized_bbox,  # type: ignore[arg-type]
                pixel_bbox=pixel_bbox,  # type: ignore[arg-type]
            )
        )
    return blocks


def load_ocr_jsonl_state(jsonl_path: Path) -> OcrJsonlState:
    state = OcrJsonlState(detector_blocks_by_frame={}, ocr_records_by_frame={})
    if not jsonl_path.exists():
        return state

    with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict) or "frame_number" not in data:
                continue

            try:
                frame_number = int(data["frame_number"])
            except (TypeError, ValueError):
                continue

            record_type = data.get("record_type")
            if record_type == "detector":
                blocks_data = data.get("detector_blocks", data.get("blocks", []))
                state.detector_blocks_by_frame[frame_number] = parse_serialized_detector_blocks(blocks_data)
            elif record_type == "ocr" or ("record_type" not in data and "spotting_items" in data):
                state.ocr_records_by_frame[frame_number] = data

    return state


def serialize_detector_record(frame_number: int, frame_rate: float, blocks: list[TextBlock]) -> str:
    data = {
        "record_type": "detector",
        "frame_number": frame_number,
        "time": round(frame_number / frame_rate, 3),
        "detector_blocks": [
            {
                "normalized_bbox": list(block.normalized_bbox),
                "pixel_bbox": list(block.pixel_bbox),
            }
            for block in blocks
        ],
    }
    return json.dumps(data, ensure_ascii=False) + "\n"


def serialize_ocr_record(
    frame_number: int,
    frame_rate: float,
    items: list[SpottingItem],
    raw_llm_output: str | None = None,
    ocr_mode: str | None = None,
    ocr_area: list[int] | None = None,
    mask_area: list[int] | None = None,
) -> str:
    data: dict[str, Any] = {
        "record_type": "ocr",
        "frame_number": frame_number,
        "time": round(frame_number / frame_rate, 3),
        "spotting_items": [item.to_dict() for item in items],
    }
    if raw_llm_output is not None:
        data["raw_llm_output"] = raw_llm_output
    if ocr_mode is not None:
        data["ocr_mode"] = ocr_mode
    if ocr_area is not None:
        data["ocr_area"] = ocr_area
    if mask_area is not None:
        data["mask_area"] = mask_area
    return json.dumps(data, ensure_ascii=False) + "\n"


def build_ocr_context(
    x: int,
    y: int,
    width: int,
    height: int,
    full_screen_ocr: bool,
    mask_x: int | None = None,
    mask_y: int | None = None,
    mask_width: int | None = None,
    mask_height: int | None = None,
) -> dict[str, Any]:
    has_mask = all(value is not None for value in (mask_x, mask_y, mask_width, mask_height))
    return {
        "ocr_mode": "full_screen" if full_screen_ocr else "crop",
        "ocr_area": [int(x), int(y), int(width), int(height)],
        "mask_area": (
            [int(mask_x), int(mask_y), int(mask_width), int(mask_height)]
            if has_mask
            else None
        ),
    }


def is_compatible_ocr_record(record: dict[str, Any], ocr_context: dict[str, Any]) -> bool:
    record_mode = record.get("ocr_mode")
    context_mode = ocr_context["ocr_mode"]
    if record_mode is None:
        return context_mode == "full_screen"
    if record_mode != context_mode:
        return False
    return (
        record.get("ocr_area") == ocr_context["ocr_area"] and
        record.get("mask_area") == ocr_context["mask_area"]
    )


def is_full_screen_ocr_record(record: dict[str, Any]) -> bool:
    record_mode = record.get("ocr_mode")
    return record_mode is None or record_mode == "full_screen"


def full_frame_text_block(image_width: int, image_height: int) -> TextBlock:
    return TextBlock(
        normalized_bbox=(0, 0, 1000, 1000),
        pixel_bbox=(0, 0, max(0, image_width - 1), max(0, image_height - 1)),
    )


def compact_ocr_jsonl(jsonl_path: Path, frame_numbers: Iterable[int]) -> None:
    state = load_ocr_jsonl_state(jsonl_path)
    temp_path = jsonl_path.with_name(f"{jsonl_path.name}.compact.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as compact_file:
        for frame_number in frame_numbers:
            record = state.ocr_records_by_frame.get(frame_number)
            if record is None:
                continue
            data: dict[str, Any] = {
                "frame_number": int(record["frame_number"]),
                "time": record.get("time", 0),
                "spotting_items": record.get("spotting_items", []),
            }
            if "raw_llm_output" in record:
                data["raw_llm_output"] = record["raw_llm_output"]
            for key in ("ocr_mode", "ocr_area", "mask_area"):
                if key in record:
                    data[key] = record[key]
            compact_file.write(json.dumps(data, ensure_ascii=False) + "\n")
    temp_path.replace(jsonl_path)


def get_ocr_frame_range(video_path: str, start_time=0, end_time=None) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video file: {video_path}")
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * frame_rate)
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if end_time is None else int(end_time * frame_rate)
    total_frames = end_frame - start_frame
    cap.release()
    return start_frame, end_frame, total_frames, frame_rate


def is_detector_cache_complete(
    video_filename: str,
    start_time=0,
    end_time=None,
) -> bool:
    paths = get_ocr_paths(video_filename)
    if not paths.jsonl_path.exists():
        return False

    start_frame, end_frame, _total_frames, _frame_rate = get_ocr_frame_range(
        str(paths.video_path),
        start_time,
        end_time,
    )
    required_frame_numbers = get_required_frame_numbers(start_frame, end_frame)
    if not required_frame_numbers:
        return True

    state = load_ocr_jsonl_state(paths.jsonl_path)
    full_screen_ocr_frame_numbers = {
        frame_number
        for frame_number, record in state.ocr_records_by_frame.items()
        if is_full_screen_ocr_record(record)
    }
    completed_frame_numbers = set(state.detector_blocks_by_frame) | full_screen_ocr_frame_numbers
    return required_frame_numbers.issubset(completed_frame_numbers)

# 시간을 포맷팅
# 동영상 파일에서 프레임을 배치 단위로 생성하는 제너레이터.
def frame_batch_generator(
    cap: cv2.VideoCapture,
    x, y, width, height,
    full_screen_ocr: bool = False,
    end_frame: int = None,
    mask_x: int | None = None,
    mask_y: int | None = None,
    mask_width: int | None = None,
    mask_height: int | None = None,
) -> Generator[List, None, None]:
    has_mask = all(value is not None for value in (mask_x, mask_y, mask_width, mask_height))
    if has_mask and not full_screen_ocr:
        raise ValueError("마스킹은 Full-screen OCR에서만 사용할 수 있습니다.")

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 지정된 종료 프레임까지 도달하면 종료
        if not ret or frame_number > end_frame:
            cap.release()
            break
        
        # 이미지 크롭
        if full_screen_ocr:
            working_frame = frame
        else:
            working_frame = frame[y:y+height, x:x+width]

        # Full-screen OCR에서 제외할 영역을 검은색으로 덮습니다.
        if has_mask:
            mask_offset_x = 0
            mask_offset_y = 0
            mask_left = max(mask_x - mask_offset_x, 0)
            mask_top = max(mask_y - mask_offset_y, 0)
            mask_right = min(mask_x + mask_width - mask_offset_x, working_frame.shape[1])
            mask_bottom = min(mask_y + mask_height - mask_offset_y, working_frame.shape[0])
            if mask_left < mask_right and mask_top < mask_bottom:
                cv2.rectangle(
                    working_frame,
                    (mask_left, mask_top),
                    (mask_right, mask_bottom),
                    color=(0, 0, 0),
                    thickness=-1,
                )

        yield [frame_number, working_frame, int(working_frame.shape[1]), int(working_frame.shape[0])]


def open_video_at_frame(video_path: str, start_frame: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video file: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    return cap

async def process_ocr(
    video_filename,
    x, y, width, height,
    start_time=0,
    end_time=None,
    full_screen_ocr=True,
    mask_x=None,
    mask_y=None,
    mask_width=None,
    mask_height=None,
    switch_to_recognizer: Callable[[], Awaitable[bool]] | None = None,
    task_id: str | None = None,
):
    has_any_mask_value = any(value is not None for value in (mask_x, mask_y, mask_width, mask_height))
    if has_any_mask_value and not full_screen_ocr:
        raise ValueError("마스킹는 Full-screen OCR에서만 사용할 수 있습니다.")
    uses_detector = full_screen_ocr

    # 파일 경로 정보 초기화
    paths = get_ocr_paths(video_filename)
    video_path = str(paths.video_path)
    jsonl_path_obj = paths.jsonl_path
    ocr_context = build_ocr_context(
        x,
        y,
        width,
        height,
        full_screen_ocr,
        mask_x,
        mask_y,
        mask_width,
        mask_height,
    )

    # 영상 정보 추출
    start_frame, end_frame, total_frames, frame_rate = get_ocr_frame_range(video_path, start_time, end_time)
    log_ocr_event(
        "pipeline",
        (
            f"영상 범위 확인: video={video_filename}, frames={start_frame + 1}~{end_frame}, "
            f"total={total_frames}, fps={frame_rate:.3f}, detector={uses_detector}"
        ),
        task_id,
    )

    # OCR 을 진행할 프레임으로 이동
    total_frames = max(1, total_frames)
    jsonl_path_obj.touch(exist_ok=True)

    # vLLM 클라이언트 초기화
    settings = get_settings()
    recognizer_client = PaddleOCRRecognizerClient(
        base_url=settings.recognizer_llm_base_url,
        model=settings.recognizer_llm_model,
        api_key=getattr(settings, "llm_api_key", None) or "dummy_key",
        log_component="recognizer-http",
        task_id=task_id,
    )

    jsonl_state = load_ocr_jsonl_state(jsonl_path_obj)
    detected_blocks_by_frame = jsonl_state.detector_blocks_by_frame
    compatible_ocr_records_by_frame = {
        frame_number: record
        for frame_number, record in jsonl_state.ocr_records_by_frame.items()
        if is_compatible_ocr_record(record, ocr_context)
    }
    ocr_frame_numbers = set(compatible_ocr_records_by_frame)
    required_frame_numbers = get_required_frame_numbers(start_frame, end_frame)
    recognizer_progress_start = 50.0 if uses_detector else 0.0

    if uses_detector:
        detector_done_frame_numbers = set(detected_blocks_by_frame) | ocr_frame_numbers
        processed_detector_frames = len(required_frame_numbers & detector_done_frame_numbers)
        detector_cache_complete = required_frame_numbers.issubset(detector_done_frame_numbers)
    else:
        processed_detector_frames = 0
        detector_cache_complete = True

    log_ocr_event(
        "pipeline",
        (
            f"캐시 로드 완료: detector_frames={len(detected_blocks_by_frame)}, "
            f"ocr_frames={len(ocr_frame_numbers)}, detector_complete={detector_cache_complete}"
        ),
        task_id,
    )

    pending_detector_tasks: set[asyncio.Task[tuple[int, list[TextBlock], str | None]]] = set()
    if uses_detector and detector_cache_complete:
        log_ocr_event("detector", "Detector 단계 건너뜀: 캐시 완료", task_id)
        yield 50
    elif uses_detector:
        detector_phase_started_at = time.monotonic()
        submitted_detector_requests = 0
        log_ocr_event(
            "detector",
            (
                f"단계 시작: model={settings.detector_llm_model}, "
                f"cached={processed_detector_frames}/{total_frames}"
            ),
            task_id,
        )
        detector_client = SuryaDetectorClient(
            base_url=settings.detector_llm_base_url,
            model=settings.detector_llm_model,
            api_key=getattr(settings, "llm_api_key", None) or "dummy_key",
            log_component="detector-http",
            task_id=task_id,
        )
        # vLLM 대기열이 비면 요청 창을 늘리고 목표 대기열을 넘으면 줄입니다.
        detector_metrics_client = VllmMetricsClient(
            base_url=settings.detector_llm_base_url or "",
            api_key=getattr(settings, "llm_api_key", None),
        )
        detector_concurrency_controller = AdaptiveConcurrencyController(
            read_load=detector_metrics_client.read_load,
            min_limit=DETECTOR_MIN_CONCURRENCY,
            initial_limit=DETECTOR_CONCURRENCY,
            max_limit=DETECTOR_MAX_CONCURRENCY,
            target_waiting=VLLM_TARGET_WAITING,
            poll_interval_sec=VLLM_METRICS_POLL_INTERVAL_SEC,
            enabled=VLLM_ADAPTIVE_CONCURRENCY,
            component="detector-load",
            task_id=task_id,
        )
        detector_cap = open_video_at_frame(video_path, start_frame)
        detector_metrics_monitor = asyncio.create_task(detector_concurrency_controller.monitor())
        try:
            yield 1
            with jsonl_path_obj.open("a", newline="", encoding="utf-8") as jsonl_file:
                async def flush_completed_detector_tasks(
                    done_tasks: set[asyncio.Task[tuple[int, list[TextBlock], str | None]]],
                ) -> list[float]:
                    nonlocal processed_detector_frames
                    progress_values: list[float] = []
                    for task in done_tasks:
                        frame_number, blocks, _ = await task
                        detected_blocks_by_frame[frame_number] = blocks
                        jsonl_file.writelines(serialize_detector_record(frame_number, frame_rate, blocks))
                        jsonl_file.flush()
                        processed_detector_frames += 1
                        progress_values.append(phase_progress(processed_detector_frames, total_frames, 0.0, 50.0, 50.0))
                    return progress_values

                for frame_idx, frame, image_width, image_height in frame_batch_generator(
                    detector_cap,
                    x,
                    y,
                    width,
                    height,
                    full_screen_ocr=full_screen_ocr,
                    end_frame=end_frame,
                    mask_x=mask_x,
                    mask_y=mask_y,
                    mask_width=mask_width,
                    mask_height=mask_height,
                ):
                    if frame_idx in detected_blocks_by_frame or frame_idx in ocr_frame_numbers:
                        yield phase_progress(processed_detector_frames, total_frames, 0.0, 50.0, 50.0)
                        continue

                    pending_detector_tasks, progress_values = await wait_for_submission_slot(
                        pending_detector_tasks,
                        flush_completed_detector_tasks,
                        detector_concurrency_controller,
                    )
                    for progress in progress_values:
                        yield progress
                    if submitted_detector_requests == 0:
                        elapsed = time.monotonic() - detector_phase_started_at
                        log_ocr_event(
                            "detector",
                            (
                                f"첫 요청 제출: frame={frame_idx}, elapsed={elapsed:.1f}초, "
                                f"limit={detector_concurrency_controller.limit}"
                            ),
                            task_id,
                        )
                    pending_detector_tasks.add(asyncio.create_task(detector_client.detect(frame_idx, frame)))
                    submitted_detector_requests += 1

                while pending_detector_tasks:
                    pending_detector_tasks, progress_values = await collect_completed_progress(
                        pending_detector_tasks,
                        flush_completed_detector_tasks,
                    )
                    for progress in progress_values:
                        yield progress
        finally:
            detector_metrics_monitor.cancel()
            await asyncio.gather(detector_metrics_monitor, return_exceptions=True)
            for task in pending_detector_tasks:
                task.cancel()
            if detector_cap.isOpened():
                detector_cap.release()
            elapsed = time.monotonic() - detector_phase_started_at
            log_ocr_event(
                "detector",
                (
                    f"단계 종료: submitted={submitted_detector_requests}, "
                    f"processed={processed_detector_frames}/{total_frames}, elapsed={elapsed:.1f}초"
                ),
                task_id,
            )

    if uses_detector and switch_to_recognizer is not None:
        log_ocr_event("pipeline", "Recognizer 전환 콜백 호출", task_id)
        should_continue = await switch_to_recognizer()
        if not should_continue:
            return
    yield 51 if uses_detector else 1

    # JSONL 파일에 OCR 결과를 저장하면 진행
    recognizer_cap = open_video_at_frame(video_path, start_frame)
    processed_recognizer_frames = len(required_frame_numbers & ocr_frame_numbers)
    pending_recognizer_tasks: set[asyncio.Task[tuple[int, int, SpottingItem | None]]] = set()
    recognizer_frame_order: list[int] = []
    recognizer_frame_states: dict[int, RecognizerFrameState] = {}
    next_recognizer_write_index = 0
    recognizer_phase_started_at = time.monotonic()
    submitted_recognizer_requests = 0
    completed_recognizer_requests = 0
    log_ocr_event(
        "recognizer",
        (
            f"단계 시작: model={settings.recognizer_llm_model}, "
            f"cached={processed_recognizer_frames}/{total_frames}"
        ),
        task_id,
    )
    # Recognizer도 별도 vLLM 모델 특성에 맞춰 독립적으로 요청 창을 조절합니다.
    recognizer_metrics_client = VllmMetricsClient(
        base_url=settings.recognizer_llm_base_url or "",
        api_key=getattr(settings, "llm_api_key", None),
    )
    recognizer_concurrency_controller = AdaptiveConcurrencyController(
        read_load=recognizer_metrics_client.read_load,
        min_limit=RECOGNIZER_MIN_CONCURRENCY,
        initial_limit=RECOGNIZER_CONCURRENCY,
        max_limit=RECOGNIZER_MAX_CONCURRENCY,
        target_waiting=VLLM_TARGET_WAITING,
        poll_interval_sec=VLLM_METRICS_POLL_INTERVAL_SEC,
        enabled=VLLM_ADAPTIVE_CONCURRENCY,
        component="recognizer-load",
        task_id=task_id,
    )
    recognizer_metrics_monitor = asyncio.create_task(recognizer_concurrency_controller.monitor())
    try:
        with jsonl_path_obj.open("a", newline="", encoding="utf-8") as jsonl_file:
            async def recognize_crop(
                target_frame_idx: int,
                block_index: int,
                block: TextBlock,
                crop,
            ) -> tuple[int, int, SpottingItem | None]:
                text = await recognizer_client.recognize(target_frame_idx, crop)
                return target_frame_idx, block_index, spotting_item_from_block(block, text)

            async def flush_completed_recognizer_tasks(
                done_tasks: set[asyncio.Task[tuple[int, int, SpottingItem | None]]],
            ) -> list[float]:
                nonlocal completed_recognizer_requests
                for task in done_tasks:
                    frame_number, block_index, spotting_item = await task
                    frame_state = recognizer_frame_states[frame_number]
                    if spotting_item is not None:
                        frame_state.items[block_index] = spotting_item
                    frame_state.remaining -= 1
                    completed_recognizer_requests += 1
                return write_ready_recognizer_frames()

            def write_ready_recognizer_frames() -> list[float]:
                nonlocal next_recognizer_write_index, processed_recognizer_frames
                progress_values: list[float] = []
                while next_recognizer_write_index < len(recognizer_frame_order):
                    frame_number = recognizer_frame_order[next_recognizer_write_index]
                    frame_state = recognizer_frame_states[frame_number]
                    if frame_state.remaining > 0:
                        break

                    spotting_items = [item for item in frame_state.items if item is not None]
                    jsonl_file.writelines(serialize_ocr_record(
                        frame_number,
                        frame_rate,
                        spotting_items,
                        ocr_mode=ocr_context["ocr_mode"],
                        ocr_area=ocr_context["ocr_area"],
                        mask_area=ocr_context["mask_area"],
                    ))
                    jsonl_file.flush()
                    ocr_frame_numbers.add(frame_number)
                    processed_recognizer_frames += 1
                    progress_values.append(
                        phase_progress(
                            processed_recognizer_frames,
                            total_frames,
                            recognizer_progress_start,
                            100.0,
                            99.0,
                        )
                    )
                    del recognizer_frame_states[frame_number]
                    next_recognizer_write_index += 1
                return progress_values

            for frame_idx, frame, image_width, image_height in frame_batch_generator(
                recognizer_cap,
                x,
                y,
                width,
                height,
                full_screen_ocr=full_screen_ocr,
                end_frame=end_frame,
                mask_x=mask_x,
                mask_y=mask_y,
                mask_width=mask_width,
                mask_height=mask_height,
            ):
                if frame_idx in ocr_frame_numbers:
                    yield phase_progress(
                        processed_recognizer_frames,
                        total_frames,
                        recognizer_progress_start,
                        100.0,
                        99.0,
                    )
                    continue

                if uses_detector:
                    blocks = detected_blocks_by_frame.get(frame_idx, [])
                else:
                    blocks = [full_frame_text_block(image_width, image_height)]
                recognizer_frame_order.append(frame_idx)
                recognizer_frame_states[frame_idx] = RecognizerFrameState(
                    remaining=0,
                    items=[None] * len(blocks),
                )

                for block_index, block in enumerate(blocks):
                    crop = crop_with_padding(frame, block.pixel_bbox)
                    if crop is None:
                        continue

                    # 슬롯 대기 중 현재 프레임이 완료된 것으로 기록되지 않도록 작업 수를 먼저 예약합니다.
                    recognizer_frame_states[frame_idx].remaining += 1
                    pending_recognizer_tasks, progress_values = await wait_for_submission_slot(
                        pending_recognizer_tasks,
                        flush_completed_recognizer_tasks,
                        recognizer_concurrency_controller,
                    )
                    for progress in progress_values:
                        yield progress
                    if submitted_recognizer_requests == 0:
                        elapsed = time.monotonic() - recognizer_phase_started_at
                        log_ocr_event(
                            "recognizer",
                            (
                                f"첫 요청 제출: frame={frame_idx}, block={block_index}, "
                                f"elapsed={elapsed:.1f}초, limit={recognizer_concurrency_controller.limit}"
                            ),
                            task_id,
                        )
                    pending_recognizer_tasks.add(
                        asyncio.create_task(recognize_crop(frame_idx, block_index, block, crop))
                    )
                    submitted_recognizer_requests += 1

                for progress in write_ready_recognizer_frames():
                    yield progress

            while pending_recognizer_tasks:
                pending_recognizer_tasks, progress_values = await collect_completed_progress(
                    pending_recognizer_tasks,
                    flush_completed_recognizer_tasks,
                )
                for progress in progress_values:
                    yield progress
            jsonl_file.flush()
    finally:
        recognizer_metrics_monitor.cancel()
        await asyncio.gather(recognizer_metrics_monitor, return_exceptions=True)
        for task in pending_recognizer_tasks:
            task.cancel()
        if recognizer_cap.isOpened():
            recognizer_cap.release()
        elapsed = time.monotonic() - recognizer_phase_started_at
        log_ocr_event(
            "recognizer",
            (
                f"단계 종료: submitted={submitted_recognizer_requests}, "
                f"completed={completed_recognizer_requests}, "
                f"frames={processed_recognizer_frames}/{total_frames}, elapsed={elapsed:.1f}초"
            ),
            task_id,
        )

    postprocess_started_at = time.monotonic()
    log_ocr_event("postprocess", "JSONL 압축 및 SRT 생성 시작", task_id)
    compact_ocr_jsonl(jsonl_path_obj, sorted(required_frame_numbers))
    jsonl_to_srt(jsonl_path_obj)
    log_ocr_event(
        "postprocess",
        f"JSONL 압축 및 SRT 생성 완료: elapsed={time.monotonic() - postprocess_started_at:.1f}초",
        task_id,
    )

    # 진행 상황 100%로 업데이트
    yield 100
