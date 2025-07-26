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
import openai
from pydantic import BaseModel, ValidationError
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import random

from core.merging_module import normalize_text, merge_ocr_texts

UPLOAD_DIR = "uploads"

base_url: str | None = os.getenv("LLM_BASE_URL")
llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")

system_prompt = """
OCR all the text from image following JSON:
{"texts":"example"}

If there is no other text from image then:
{"texts":""}
"""

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
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

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
async def ocr_one_frame(
    client: openai.AsyncOpenAI,
    frame_number: int,
    img_base64: str,
) -> tuple[int, str]:            # (frame_number, ocr_text) 반환
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
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
    except (ValidationError, json.JSONDecodeError) as e:
        # 예외가 발생해도 그냥 빈 문자열로 치환하고 로그만 남긴다
        print(f"[Warn] 프레임 {frame_number} OCR 중 예외 발생: {e!r}")
        ocr_text = ""
    except Exception as e:
        # 혹시 다른 예외도 잡고 싶으면 여기에 추가
        print(f"[Error] 예기치 못한 오류 (frame {frame_number}): {e!r}")
        ocr_text = ""
    return frame_number, ocr_text
    
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
        raise ValueError(f"Cannot open video file: {video_path}")
    frame_rate = cap.get(cv2.CAP_PROP_FPS) # 초당 프레임 수
    start_frame = int(start_time * frame_rate) # OCR 시작 프레임
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if end_time is None else int(end_time * frame_rate) # OCR 종료 프레임
    total_frames = end_frame - start_frame # OCR 을 진행할 총 프레임 수


    # OCR 을 진행할 프레임으로 이동 
    start_ocr_frame = max(start_frame, last_frame_number)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_ocr_frame) 

    # ollama 클라이언트 초기화
    client = openai.AsyncOpenAI(base_url=base_url, api_key="dummy_key")

    # vllm 서버가 실행 중인지 확인
    try:
        await client.models.list()
    except openai.APIConnectionError:
        raise RuntimeError("vllm 서버가 실행 중인지 확인해주세요.")

    # CSV 파일에 OCR 결과를 저장하면 진행
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['frame_number', 'time', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        #  결과를 순서대로 내보내기 위한 우선순위 큐
        heap: list[tuple[int, str]] = []
        next_frame_to_write = start_ocr_frame + 1

        #  실행 중·대기 중인 태스크 집합
        running: set[asyncio.Task] = set()

        for frame_number, img_b64 in frame_batch_generator(cap, x, y, width, height, end_frame):

            # 새 태스크 추가
            task = asyncio.create_task(ocr_one_frame(client, frame_number, img_b64))
            running.add(task)

            #  태스크 수가 N를 초과하지 않도록 완료될 때까지 대기
            if len(running) >= 3:
                done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)

                for t in done:
                    fn, txt = t.result()
                    heappush(heap, (fn, txt))
                
            #  heap 안에 다음 프레임이 있으면 순서대로 기록
            while heap and heap[0][0] == next_frame_to_write:
                fn, ocr_text = heappop(heap)
                ocr_text = normalize_text(ocr_text)  # 불필요한 대괄호 제거
                if ocr_text:
                    print(ocr_text)
                writer.writerow({
                    "frame_number": fn,
                    "time": round(fn / frame_rate, 3),
                    "text": ocr_text,
                })
                next_frame_to_write += 1
                percentage = round((fn - start_frame) / total_frames * 100, 2)
                yield percentage
            csvfile.flush()
        
        # 남아 있는 태스크 모두 완료
        for t in asyncio.as_completed(running):
            fn, txt = await t
            heappush(heap, (fn, txt))

        # heap 잔여 결과 정리
        while heap:
            fn, ocr_text = heappop(heap)
            writer.writerow({
                "frame_number": fn,
                "time": round(fn / frame_rate, 3),
                "text": ocr_text,
            })
        csvfile.flush()
        
        # 진행 상황 업데이트
        yield 100

    # OCR 완료된 CSV 파일 읽기
    ocr_text_data = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        for row in csv.DictReader(csvfile):
            ocr_text_data.append({
                'frame_number': row['frame_number'],
                'time': float(row['time']),
                'text': row['text'],
            })

    # 언어 감지 및 자막 저장 경로 설정
    langs = []
    non_empty_entries = [entry for entry in ocr_text_data if entry['text'].strip()]
    
    # Only proceed with sampling if there are non-empty entries
    if non_empty_entries:
        sample_size = min(100, len(non_empty_entries))
        samples = random.sample(non_empty_entries, sample_size)
        
        for entry in samples:
            text_content = entry['text'].strip()
            try:
                lang_detected = detect(text_content)
                langs.append(lang_detected)
            except LangDetectException:
                pass
    most_common_lang = Counter(langs).most_common(1)[0][0] if langs else "un"
    srt_path = str(video_path_obj.parent / (video_path_obj.with_suffix('').name + f".{most_common_lang}.srt"))

    # 텍스트 병합 모듈 사용
    ocr_progress_data = merge_ocr_texts(ocr_text_data)

    # 자막 파일 생성
    with open(srt_path, 'w', encoding='utf-8') as f:
        for idx, subtitle in enumerate(ocr_progress_data, start=1):
            start = format_time(subtitle.start_time)
            end = format_time(subtitle.end_time)
            subtitle_line = subtitle.text.replace("\\n", "\n")
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{subtitle_line}\n\n")

    # 진행 상황 100%로 업데이트
    yield 100
