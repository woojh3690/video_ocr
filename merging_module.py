# text_merging_module.py
from thefuzz import fuzz
import re

def is_valid_text(text):
    # 텍스트가 None이거나 빈 문자열인 경우
    if not text or not text.strip():
        return False

    stripped_text = text.strip()

    # 텍스트 길이 기준 (3자 미만인 경우 필터링)
    if len(stripped_text) < 3:
        return False
    
    # 특수 문자 또는 숫자만 있는 경우
    if re.fullmatch(r'[\d\s%:/.\-+,]+', text):
        return False

    # 반복되는 패턴 검사
    if re.fullmatch(r'(.)\1{2,}', stripped_text):
        return False

    return True

def merge_ocr_texts(ocr_text_data, similarity_threshold=70):
    ocr_progress_data = []
    current_subtitle = None

    for entry in ocr_text_data:
        current_time = entry['time']
        ocr_text = entry['text']

        # 텍스트 필터링 적용
        if not is_valid_text(ocr_text):
            continue  # 유효하지 않은 텍스트는 건너뜁니다.

        if current_subtitle is None:
            current_subtitle = {
                'start_time': current_time,
                'end_time': current_time,
                'text': ocr_text
            }
        else:
            similarity = fuzz.ratio(current_subtitle['text'], ocr_text)
            if similarity > similarity_threshold:
                current_subtitle['end_time'] = current_time
                current_subtitle['text'] = ocr_text
            else:
                ocr_progress_data.append(current_subtitle)
                current_subtitle = {
                    'start_time': current_time,
                    'end_time': current_time,
                    'text': ocr_text
                }
    if current_subtitle:
        ocr_progress_data.append(current_subtitle)

    return ocr_progress_data
