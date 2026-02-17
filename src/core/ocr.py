import os
import csv
import json
import base64
import asyncio
from pathlib import Path
from heapq import heappush, heappop
from typing import List, Generator
from collections import Counter

import cv2

from core.paddle_client import OcrProcessingError, SpottingItem, PaddleClient
from core.jsonl_to_srt import jsonl_to_srt
from core.settings_manager import get_settings

UPLOAD_DIR = "uploads"

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

        # 이미지를 base64로 인코딩
        rgb_frame = cv2.cvtColor(working_frame, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode('.jpg', rgb_frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        yield [frame_number, img_base64]

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
):
    has_any_mask_value = any(value is not None for value in (mask_x, mask_y, mask_width, mask_height))
    if has_any_mask_value and not full_screen_ocr:
        raise ValueError("Mask is only supported when full_screen_ocr is enabled.")

    # 파일 경로 정보 초기화
    UPLOAD_DIR = "uploads"
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    video_path_obj = Path(video_path)
    jsonl_path_obj = video_path_obj.with_suffix(".jsonl")

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

    def write_json(fn: int, items: list[SpottingItem]):
        ocr_res_dict = {
            "frame_number": fn,
            "time": round(frame_number / frame_rate, 3),
            "spotting_items": [item.to_dict() for item in items],
        }
        line = json.dumps(ocr_res_dict, ensure_ascii=False) + "\n"
        jsonl_file.writelines(line)

    # OCR 을 진행할 프레임으로 이동 
    start_ocr_frame = max(start_frame, last_frame_number)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_ocr_frame)

    # ollama 클라이언트 초기화
    settings = get_settings()
    client = PaddleClient(base_url=settings.llm_base_url, model=settings.llm_model)
    
    running: set[asyncio.Task] = set()

    # CSV 파일에 OCR 결과를 저장하면 진행
    jsonl_file = jsonl_path_obj.open("a", newline="", encoding="utf-8")
    try:
        #  결과를 순서대로 내보내기 위한 우선순위 큐
        heap: list[tuple[int, str]] = []
        next_frame_to_write = start_ocr_frame + 1

        for frame_idx, img_b64 in frame_batch_generator(
            cap,
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
            # 새 작업 추가
            running.add(
                asyncio.create_task(
                    client.predict(frame_idx, img_b64)
                )
            )

            #  작업 수가 N개를 초과하였을 경우 완료된 작업만 기록
            if len(running) >= 8:
                done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    try:
                        heappush(heap, t.result())
                    except OcrProcessingError as exc:
                        raise exc
            
            #  heap 안에 다음 프레임이 있으면 순서대로 기록
            while heap and heap[0][0] == next_frame_to_write:
                frame_number, spotting_items = heappop(heap)
                write_json(frame_number, spotting_items)
                next_frame_to_write += 1
                yield round((frame_number - start_frame) / total_frames * 100, 2)
            jsonl_file.flush()
        
        # 남아 있는 태스크 모두 완료
        if running:
            results = await asyncio.gather(*running, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    raise result
                heappush(heap, result)

        # heap 잔여 결과 정리
        while heap:
            frame_number, spotting_items = heappop(heap)
            write_json(frame_number, spotting_items)
        jsonl_file.flush()
    finally:
        if running:
            for task in list(running):
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            running.clear()
        jsonl_file.close()
        if cap.isOpened():
            cap.release()

    jsonl_to_srt(jsonl_path_obj)

    # 진행 상황 100%로 업데이트
    yield 100
