import os
import csv
import json
import base64
from typing import List, Generator

import cv2
import ollama
from pydantic import BaseModel, ValidationError

from merging_module import merge_ocr_texts  # 모듈 임포트

system_prompt = 'You are a ocr assistant. Extract all the subtitles and text in JSON format. \
Group the subtitles according to color of the subtitles; \
subtitles with the same color belong to the same group.\n\
Use the following JSON format: {\"ocr_subtitles_group\":[[\"group1 first subtitle\",\"group1 second subtitle\",\"...\"],\
[\"group2 first subtitle\",\"group2 second subtitle\",\"...\"]]}\n\
If there is no subtitles then: {\"ocr_subtitles_group\":[]}'

class OcrSubtitleGroup(BaseModel):
    ocr_subtitles_group: list[list[str]]

# 동영상 파일에서 프레임을 배치 단위로 생성하는 제너레이터.
def frame_batch_generator(
    cap: cv2.VideoCapture, 
    last_frame_number: int, 
    x, y, width, height
) -> Generator[List, None, None]:
    last_frame_number = -1 if last_frame_number is None else last_frame_number

    interval = 0.3  # 0.3초마다 OCR 수행
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(frame_rate * interval)))  # 최소 1프레임

    frame_number = -1
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break
        frame_number += 1

        # 처리된 않은 프레임 부터 OCR 진행
        if frame_number <= last_frame_number:
            continue
        
        # 0.3초마다 OCR 수행
        if frame_number % frame_interval != 0:
            continue

        # 이미지 크롭
        cropped_frame = frame[y:y+height, x:x+width]

        # 이미지를 base64로 인코딩
        rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode('.jpg', rgb_frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        yield [frame_number, img_base64]

# 가장 마지막으로 OCR 처리된 프레임 번호를 반환
def get_last_processed_frame_number(csv_path) -> tuple[int, List]:
    # 기존 OCR 데이터를 로드하여 진행 상황을 파악
    last_frame_number = None

    # 파일이 존재하지 않으면 초기 상태로 반환
    if not os.path.exists(csv_path):
        return last_frame_number
    
    # 파일이 존재하면 마지막 프레임 번호와 OCR 데이터를 로드
    last_frame_number = -1
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
    
async def process_ocr(video_filename, x, y, width, height):
    # 파일 경로 정보 초기화
    UPLOAD_DIR = "uploads"
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    filename_without_ext = os.path.splitext(os.path.basename(video_filename))[0]
    csv_path = os.path.join(UPLOAD_DIR, f"{filename_without_ext}.csv")
    srt_path = os.path.join(UPLOAD_DIR, f"{filename_without_ext}.srt")

    # 영상 정보 추출
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # ollama 클라이언트 초기화
    client = ollama.AsyncClient()

    # CSV 파일을 열어둔 채로 진행
    last_frame_number = get_last_processed_frame_number(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['frame_number', 'time', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if last_frame_number == None:
            writer.writeheader()
            csvfile.flush()

        for frame in frame_batch_generator(cap, last_frame_number, x, y, width, height):
            frame_number = frame[0]
            img_base64 = frame[1]

            # OCR 수행
            response = await client.chat(
                model='minicpm-v',
                format=OcrSubtitleGroup.model_json_schema(),
                options={'temperature': 0},
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt,
                    },
                    {
                        'role': 'user',
                        'images': [img_base64],
                    }
                ],
            )
            content = response.message.content
            try:
                ocr_subtitles_group = OcrSubtitleGroup.model_validate_json(content).ocr_subtitles_group
            except ValidationError:
                print(f"{frame_number} 프레임 JSON 디코딩 에러:")
                print(content)
                continue
            
            # 후처리
            ## 줄 병합
            merged_subtitle = []
            for subtitles in ocr_subtitles_group:
                merged_subtitle.append("\n".join(subtitles))
            ocr_text = "\n\n".join(merged_subtitle).strip()

            if ocr_text:
                print(ocr_text)

            ## OCR 결과가 없는 경우 처리
            if any(phrase in ocr_text for phrase in [
                "image does not contain any", 
                "There is no visible text in this image."
            ]):
                ocr_text = ""
            
            ## 줄바꿈 문자 이스케이프 처리
            ocr_text = ocr_text.replace('\n', '\\n')

            # CSV 파일에 쓰기
            entry = {
                'frame_number': frame_number, 
                'time': round(frame_number / frame_rate, 3), 
                'text': ocr_text
            }
            writer.writerow(entry)

            # 진행 상황 업데이트
            percentage = round((frame_number / total_frames) * 100, 2)
            progress = f"data: {json.dumps({'progress': percentage})}\n\n"
            yield progress

    # OCR 완료된 CSV 파일 읽기기
    ocr_text_data = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        for row in csv.DictReader(csvfile):
            ocr_text_data.append({
                'frame_number': row['frame_number'],
                'time': float(row['time']),
                'text': row['text'],
            })

    # 텍스트 병합 모듈 사용
    ocr_progress_data = merge_ocr_texts(ocr_text_data)

    # 자막 파일 생성
    with open(srt_path, 'w', encoding='utf-8') as f:
        for idx, subtitle in enumerate(ocr_progress_data, start=1):
            start = format_time(subtitle['start_time'])
            end = format_time(subtitle['end_time'])
            subtitle_line = subtitle['text'].replace("\\n", "\n")
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{subtitle_line}\n\n")

    # 진행 상황 100%로 업데이트
    yield f"data: {json.dumps({'progress': 100})}\n\n"
