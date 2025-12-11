import os
import csv
import json
import base64
import asyncio
import unicodedata
from pathlib import Path
from heapq import heappush, heappop
from typing import List, Generator
from collections import Counter

import cv2
import openai
from openai import LengthFinishReasonError
from pydantic import BaseModel, ValidationError
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import random

from core.csv_to_srt import normalize_text as collapse_whitespace, convert_csv_to_srt
from core.settings_manager import get_settings

UPLOAD_DIR = "uploads"

BRACKET_CHARS = "[](){}<>「」『』〈〉《》"
BRACKET_PAIRS = {
    BRACKET_CHARS[i]: BRACKET_CHARS[i + 1]
    for i in range(0, len(BRACKET_CHARS), 2)
}

SYSTEM_PROMPT = """
OCR all the text from image following JSON:
{"texts":"example"}

If there is no other text from image then:
{"texts":""}
"""

# llm 서버에서 치명적인 오류가 발생한 경우
class OcrProcessingError(RuntimeError):
    def __init__(self, frame_number: int, original_exception: Exception):
        self.frame_number = frame_number
        self.original_exception = original_exception
        message = f"프레임 {frame_number} OCR 중 예기치 못한 오류: {original_exception}"
        super().__init__(message)

def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    
    # 노멀라이즈
    normalized = unicodedata.normalize("NFKC", text)

    # 양 끝에 있는 브라켓 제거
    while len(normalized) > 1:
        opening = normalized[0]
        closing = BRACKET_PAIRS.get(opening)
        if not closing or normalized[-1] != closing:
            break
        normalized = normalized[1:-1]
    return collapse_whitespace(normalized)

class OcrSubtitleGroup(BaseModel):
    texts: str

# 가장 마지막으로 OCR 처리된 프레임 번호를 반환
def get_last_processed_frame_number(csv_path, fieldnames) -> int:
    last_frame_number = -1
    if not os.path.exists(csv_path):
        # 파일이 존재하지 않으면 초기 상태로 반환
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            csvfile.flush()
    else:
        # 파일이 존재하면 마지막 프레임 번호와 OCR 데이터를 로드
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            for row in csv.DictReader(csvfile):
                frame_number = int(row['frame_number'])
                last_frame_number = max(last_frame_number, frame_number)
    return last_frame_number

# 시간을 포맷팅
# 동영상 파일에서 프레임을 배치 단위로 생성하는 제너레이터.
def frame_batch_generator(
    cap: cv2.VideoCapture,
    x, y, width, height,
    end_frame: int = None
) -> Generator[List, None, None]:
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 지정된 종료 프레임까지 도달하면 종료
        if not ret or frame_number > end_frame:
            cap.release()
            break
        
        # 이미지 크롭
        cropped_frame = frame[y:y+height, x:x+width]

        # 이미지를 base64로 인코딩
        rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode('.jpg', rgb_frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        yield [frame_number, img_base64]

# OCR 1회 호출을 비동기 태스크로 분리
async def ocr_one_frame(client: openai.AsyncOpenAI, frame_idx: int, 
                        img_base64: str, llm_model: str) -> tuple[int, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]}
    ]
    try:
        completion = await client.beta.chat.completions.parse(
            model=llm_model,
            messages=messages,
            response_format=OcrSubtitleGroup,
            temperature=0,
            max_tokens=512,
        )
        ocr_text = completion.choices[0].message.parsed.texts
        if len(ocr_text) == 1 or ocr_text == "example":
            ocr_text = ""
    except (ValidationError, json.JSONDecodeError, LengthFinishReasonError) as e:
        # 예외가 발생해도 그냥 빈 문자열로 치환하고 로그만 남긴다
        print(f"[Warn] 프레임 {frame_idx} OCR 중 예외 발생: {e!r}")
        ocr_text = ""
    except Exception as e:
        error = OcrProcessingError(frame_idx, e)
        print(f"[Error] {error}")
        raise error
    return frame_idx, ocr_text
    
async def process_ocr(
    video_filename,
    x, y, width, height,
    start_time=0,
    end_time=None
):
    # 파일 경로 정보 초기화
    UPLOAD_DIR = "uploads"
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    video_path_obj = Path(video_path)
    csv_path = str(video_path_obj.with_suffix(".csv"))

    # 마지막으로 OCR 처리된 프레임 번호 확인
    fieldnames = ['frame_number', 'time', 'text']
    last_frame_number = get_last_processed_frame_number(csv_path, fieldnames)

    # 영상 정보 추출
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video file: {video_path}")
    frame_rate = cap.get(cv2.CAP_PROP_FPS) # 초당 프레임 수
    start_frame = int(start_time * frame_rate) # OCR 시작 프레임
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if end_time is None else int(end_time * frame_rate) # OCR 종료 프레임
    total_frames = end_frame - start_frame # OCR 을 진행할 총 프레임 수

    # OCR 을 진행할 프레임으로 이동 
    start_ocr_frame = max(start_frame, last_frame_number)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_ocr_frame)

    # ollama 클라이언트 초기화
    settings = get_settings()
    client = openai.AsyncOpenAI(base_url=settings.llm_base_url or None, api_key="dummy_key")

    # vllm 서버가 실행 중인지 확인
    try:
        await client.models.list()
    except openai.APIConnectionError:
        raise RuntimeError("vllm 서버가 실행 중인지 확인해주세요.")
    
    running: set[asyncio.Task] = set()

    # CSV 파일에 OCR 결과를 저장하면 진행
    csvfile = open(csv_path, 'a', newline='', encoding='utf-8')
    try:
        fieldnames = ['frame_number', 'time', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        #  결과를 순서대로 내보내기 위한 우선순위 큐
        heap: list[tuple[int, str]] = []
        next_frame_to_write = start_ocr_frame + 1

        for frame_idx, img_b64 in frame_batch_generator(cap, x, y, width, height, end_frame):
            # 새 작업 추가
            running.add(
                asyncio.create_task(
                    ocr_one_frame(client, frame_idx, img_b64, settings.llm_model)
                )
            )

            #  작업 수가 N개를 초과하였을 경우 완료된 작업만 기록
            if len(running) >= 4:
                done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    try:
                        frame_number, txt = t.result()
                        heappush(heap, (frame_number, txt))
                    except OcrProcessingError as exc:
                        raise exc
            
            #  heap 안에 다음 프레임이 있으면 순서대로 기록
            while heap and heap[0][0] == next_frame_to_write:
                frame_number, ocr_text = heappop(heap)
                ocr_text = clean_ocr_text(ocr_text)  # 불필요한 대괄호 제거
                writer.writerow({
                    "frame_number": frame_number,
                    "time": round(frame_number / frame_rate, 3),
                    "text": ocr_text,
                })
                next_frame_to_write += 1
                percentage = round((frame_number - start_frame) / total_frames * 100, 2)
                if ocr_text:
                    print(f"[{frame_number}][{percentage:.2f}]: {ocr_text}")
                yield percentage
            csvfile.flush()
        
        # 남아 있는 태스크 모두 완료
        if running:
            results = await asyncio.gather(*running, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    raise result
                frame_number, txt = result
                heappush(heap, (frame_number, txt))

        # heap 잔여 결과 정리
        while heap:
            frame_number, ocr_text = heappop(heap)
            ocr_text = clean_ocr_text(ocr_text)
            writer.writerow({
                "frame_number": frame_number,
                "time": round(frame_number / frame_rate, 3),
                "text": ocr_text,
            })
        csvfile.flush()
        
        # 진행 상황 업데이트
        yield 100
    finally:
        if running:
            for task in list(running):
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            running.clear()
        csvfile.close()
        if cap.isOpened():
            cap.release()

    # OCR 완료 후 CSV 파일 읽기
    non_empty_texts: List[str] = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        for row in csv.DictReader(csvfile):
            text_value = row['text']
            if text_value and text_value.strip():
                non_empty_texts.append(text_value.strip())

    # 언어 감지 및 자막 경로 결정
    langs: List[str] = []
    if non_empty_texts:
        sample_size = min(100, len(non_empty_texts))
        for text_content in random.sample(non_empty_texts, sample_size):
            try:
                langs.append(detect(text_content))
            except LangDetectException:
                continue
    most_common_lang = Counter(langs).most_common(1)[0][0] if langs else "un"

    csv_path_obj = Path(csv_path)
    srt_path = csv_path_obj.with_suffix(f".{most_common_lang}.srt")

    convert_csv_to_srt(csv_path_obj, srt_path)

    # 진행 상황 100%로 업데이트
    yield 100
