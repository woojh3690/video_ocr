import asyncio
import os
import json
from pathlib import Path
from typing import Awaitable, Callable, Generator, List

import cv2

from core.hunyuan_client import OcrProcessingError, SpottingItem
from core.jsonl_to_srt import jsonl_to_srt
from core.settings_manager import get_settings
from core.split_ocr_client import (
    ChandraDetectorClient,
    ChandraTextBlock,
    PaddleOCRRecognizerClient,
    crop_with_padding,
    spotting_item_from_block,
)

UPLOAD_DIR = "uploads"


def read_positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


DETECTOR_CONCURRENCY = read_positive_int_env("DETECTOR_CONCURRENCY", 8)

# 가장 마지막으로 OCR 처리된 프레임 번호를 반환
def get_last_frame_number(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
    
    with jsonl_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell() - 1

        # 파일 끝의 개행 문자를 건너뛴 뒤 마지막 라인만 읽는다.
        while pos >= 0:
            f.seek(pos)
            if f.read(1) not in (b"\n", b"\r"):
                break
            pos -= 1

        if pos >= 0:
            line_end = pos + 1
            while pos >= 0:
                f.seek(pos)
                if f.read(1) == b"\n":
                    break
                pos -= 1

            f.seek(pos + 1)
            last_line = f.read(line_end - (pos + 1)).decode("utf-8")
            last_obj = json.loads(last_line)
            return int(last_obj.get("frame_number"))
    return 0


def load_detector_cache(detector_jsonl_path: Path) -> dict[int, list[ChandraTextBlock]]:
    detected_blocks_by_frame: dict[int, list[ChandraTextBlock]] = {}
    if not detector_jsonl_path.exists():
        return detected_blocks_by_frame

    with detector_jsonl_path.open("r", encoding="utf-8") as detector_file:
        for line in detector_file:
            if not line.strip():
                continue
            data = json.loads(line)
            frame_number = int(data["frame_number"])
            blocks = [
                ChandraTextBlock(
                    normalized_bbox=tuple(block["normalized_bbox"]),  # type: ignore[arg-type]
                    pixel_bbox=tuple(block["pixel_bbox"]),  # type: ignore[arg-type]
                )
                for block in data.get("blocks", [])
            ]
            detected_blocks_by_frame[frame_number] = blocks
    return detected_blocks_by_frame


def serialize_detector_blocks(frame_number: int, blocks: list[ChandraTextBlock]) -> str:
    data = {
        "frame_number": frame_number,
        "blocks": [
            {
                "normalized_bbox": list(block.normalized_bbox),
                "pixel_bbox": list(block.pixel_bbox),
            }
            for block in blocks
        ],
    }
    return json.dumps(data, ensure_ascii=False) + "\n"

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
        raise ValueError("Mask is only supported when full_screen_ocr is enabled.")

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

        if has_mask:
            mask_left = max(mask_x, 0)
            mask_top = max(mask_y, 0)
            mask_right = min(mask_left + mask_width, working_frame.shape[1])
            mask_bottom = min(mask_top + mask_height, working_frame.shape[0])
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
    full_screen_ocr=False,
    mask_x=None,
    mask_y=None,
    mask_width=None,
    mask_height=None,
    switch_to_recognizer: Callable[[], Awaitable[bool]] | None = None,
):
    has_any_mask_value = any(value is not None for value in (mask_x, mask_y, mask_width, mask_height))
    if has_any_mask_value and not full_screen_ocr:
        raise ValueError("Mask is only supported when full_screen_ocr is enabled.")

    # 파일 경로 정보 초기화
    UPLOAD_DIR = "uploads"
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    video_path_obj = Path(video_path)
    jsonl_path_obj = video_path_obj.with_suffix(".jsonl")
    detector_jsonl_path_obj = video_path_obj.with_suffix(".detector.jsonl")

    # 마지막으로 OCR 처리된 프레임 번호 확인
    last_frame_number = get_last_frame_number(jsonl_path_obj)

    # 영상 정보 추출
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video file: {video_path}")
    frame_rate = cap.get(cv2.CAP_PROP_FPS) # 초당 프레임 수
    start_frame = int(start_time * frame_rate) # OCR 시작 프레임
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if end_time is None else int(end_time * frame_rate) # OCR 종료 프레임
    total_frames = end_frame - start_frame # OCR 을 진행할 총 프레임 수
    cap.release()

    def write_json(fn: int, items: list[SpottingItem], raw_llm_output: str | None = None):
        ocr_res_dict = {
            "frame_number": fn,
            "time": round(fn / frame_rate, 3),
            "spotting_items": [item.to_dict() for item in items],
        }
        if raw_llm_output is not None:
            ocr_res_dict["raw_llm_output"] = raw_llm_output
        line = json.dumps(ocr_res_dict, ensure_ascii=False) + "\n"
        return line

    # OCR 을 진행할 프레임으로 이동
    start_ocr_frame = max(start_frame, last_frame_number)
    total_frames = max(1, total_frames)
    jsonl_path_obj.touch(exist_ok=True)

    # vLLM 클라이언트 초기화
    settings = get_settings()
    detector_client = ChandraDetectorClient(
        base_url=settings.detector_llm_base_url,
        model=settings.detector_llm_model,
        api_key=getattr(settings, "llm_api_key", None) or "dummy_key",
    )
    recognizer_client = PaddleOCRRecognizerClient(
        base_url=settings.recognizer_llm_base_url,
        model=settings.recognizer_llm_model,
        api_key=getattr(settings, "llm_api_key", None) or "dummy_key",
    )

    detected_blocks_by_frame = load_detector_cache(detector_jsonl_path_obj)
    processed_detector_frames = len(detected_blocks_by_frame)

    detector_cap = open_video_at_frame(video_path, start_ocr_frame)
    pending_detector_tasks: set[asyncio.Task[tuple[int, list[ChandraTextBlock], str | None]]] = set()
    yield 1
    try:
        with detector_jsonl_path_obj.open("a", newline="", encoding="utf-8") as detector_file:
            async def flush_completed_detector_tasks(
                done_tasks: set[asyncio.Task[tuple[int, list[ChandraTextBlock], str | None]]],
            ) -> list[float]:
                nonlocal processed_detector_frames
                progress_values: list[float] = []
                for task in done_tasks:
                    frame_number, blocks, _ = await task
                    detected_blocks_by_frame[frame_number] = blocks
                    detector_file.writelines(serialize_detector_blocks(frame_number, blocks))
                    detector_file.flush()
                    processed_detector_frames += 1
                    progress_values.append(round(min(50.0, processed_detector_frames / total_frames * 50.0), 2))
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
                if frame_idx in detected_blocks_by_frame:
                    yield round(min(50.0, processed_detector_frames / total_frames * 50.0), 2)
                    continue

                pending_detector_tasks.add(asyncio.create_task(detector_client.detect(frame_idx, frame)))
                if len(pending_detector_tasks) >= DETECTOR_CONCURRENCY:
                    done, pending_detector_tasks = await asyncio.wait(
                        pending_detector_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for progress in await flush_completed_detector_tasks(done):
                        yield progress

            while pending_detector_tasks:
                done, pending_detector_tasks = await asyncio.wait(
                    pending_detector_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for progress in await flush_completed_detector_tasks(done):
                    yield progress
    finally:
        for task in pending_detector_tasks:
            task.cancel()
        if detector_cap.isOpened():
            detector_cap.release()

    if switch_to_recognizer is not None:
        should_continue = await switch_to_recognizer()
        if not should_continue:
            return
    yield 51

    # JSONL 파일에 OCR 결과를 저장하면 진행
    recognizer_cap = open_video_at_frame(video_path, start_ocr_frame)
    processed_recognizer_frames = 0
    try:
        with jsonl_path_obj.open("a", newline="", encoding="utf-8") as jsonl_file:
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
                spotting_items: list[SpottingItem] = []
                for block in detected_blocks_by_frame.get(frame_idx, []):
                    crop = crop_with_padding(frame, block.pixel_bbox)
                    if crop is None:
                        continue
                    text = await recognizer_client.recognize(frame_idx, crop)
                    spotting_item = spotting_item_from_block(block, text)
                    if spotting_item is not None:
                        spotting_items.append(spotting_item)

                jsonl_file.writelines(write_json(frame_idx, spotting_items))
                processed_recognizer_frames += 1
                yield round(min(99.0, 50.0 + processed_recognizer_frames / total_frames * 50.0), 2)
                jsonl_file.flush()
            jsonl_file.flush()
    finally:
        if recognizer_cap.isOpened():
            recognizer_cap.release()

    jsonl_to_srt(jsonl_path_obj)
    if detector_jsonl_path_obj.exists():
        detector_jsonl_path_obj.unlink()

    # 진행 상황 100%로 업데이트
    yield 100
