# text_merging_module.py
from thefuzz import fuzz

def merge_ocr_texts(ocr_text_data, similarity_threshold=50):
    ocr_progress_data = []
    current_subtitle = None

    for entry in ocr_text_data:
        current_time = entry['time']
        ocr_text = entry['text']

        if current_subtitle is None:
            current_subtitle = {
                'start_time': current_time,
                'end_time': current_time,
                'text': ocr_text
            }
        else:
            similarity = fuzz.partial_ratio(current_subtitle['text'], ocr_text)
            if similarity > similarity_threshold:
                current_subtitle['end_time'] = current_time
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
