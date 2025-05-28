import re
from dataclasses import dataclass

from thefuzz import fuzz

@dataclass
class Subtitle:
    start_time: float
    end_time: float
    text: str
    history: list[str]

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

def get_init_subtitle(current_time, ocr_text) -> Subtitle:
    return Subtitle(
        start_time=current_time,
        end_time=current_time,
        text=ocr_text,
        history= [ocr_text]
    )

def merge_ocr_texts(ocr_text_data, similarity_threshold=60) -> list[Subtitle]:
    ocr_progress_data: list[Subtitle] = []
    current_subtitle: Subtitle = None

    none_subtitle_interval = 0  # 이전 자막 공백 카운트
    for entry in ocr_text_data:
        current_time = entry['time']
        ocr_text = entry['text']

        # 유효하지 않은 자막은 건너뜀
        if not is_valid_text(ocr_text):
            none_subtitle_interval += 1
            continue

        if current_subtitle is None:
            current_subtitle = get_init_subtitle(current_time, ocr_text)
        else:
            # 자막 유사성 및 자막 공백 카운트를 고려하여 자막 병합
            similarity = fuzz.ratio(current_subtitle.text, ocr_text)
            if similarity > similarity_threshold and none_subtitle_interval < 8:
                current_subtitle.end_time = current_time
                current_subtitle.text = ocr_text
                current_subtitle.history.append(ocr_text)
            else:
                ocr_progress_data.append(current_subtitle)
                current_subtitle = get_init_subtitle(current_time, ocr_text)
        none_subtitle_interval = 0
    
    if current_subtitle:
        ocr_progress_data.append(current_subtitle)

    grouped: list[Subtitle] = []

    # 그룹화된 자막을 기반으로 후처리
    for subtitle in ocr_progress_data:
        # history 가 너무 적은 경우 잘못 ocr 된 것으로 판단하여 제거
        if len(subtitle.history) < 4:
            continue

        # 자막 텍스트 히스토리를 기반으로 가장 많이 등장한 텍스트를 선택
        # 가장 많이 등장한 텍스트가 여러 개인 경우 가장 먼저 나온 텍스트를 선택
        subtitle.text = max(
            enumerate(subtitle.history),
            key=lambda x: (subtitle.history.count(x[1]), x[0])  # count와 인덱스를 기준으로 정렬
        )[1]  # 최종적으로 값만 가져옴

        del subtitle.history
        grouped.append(subtitle)

    # 이전 자막과 완전히 동일하면 자막 병합
    if len(grouped) > 0:
        result: list[Subtitle] = [grouped[0]]
        for subtitle in grouped[1:]:
            if result[-1].text == subtitle.text:
                result[-1].end_time = subtitle.end_time
            else:
                result.append(subtitle)
    else:
        result = []

    return result
