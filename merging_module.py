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

    none_subtitle_interval = 0  # 이전 자막 공백 카운트
    for entry in ocr_text_data:
        current_time = entry['time']
        ocr_text = entry['text']

        # 유효하지 않은 자막은 건너뜀
        if not is_valid_text(ocr_text):
            none_subtitle_interval += 1
            continue

        if current_subtitle is None:
            current_subtitle = {
                'start_time': current_time,
                'end_time': current_time,
                'text': ocr_text
            }
        else:
            # 자막 유사성 및 자막 공백 카운트를 고려하여 자막 병합
            similarity = fuzz.ratio(current_subtitle['text'], ocr_text)
            if similarity > similarity_threshold and none_subtitle_interval < 8:
                current_subtitle['end_time'] = current_time
                current_subtitle['text'] = ocr_text
            else:
                ocr_progress_data.append(current_subtitle)
                current_subtitle = {
                    'start_time': current_time,
                    'end_time': current_time,
                    'text': ocr_text
                }
        none_subtitle_interval = 0
    
    if current_subtitle:
        ocr_progress_data.append(current_subtitle)

    return ocr_progress_data
