import os
import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from merging_module import merge_ocr_texts

def test_merge_ocr_texts_from_csv(csv_filename):
    if csv_filename == "": return
    
    ocr_text_data = []
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ocr_text_data.append({'time': float(row['time']), 'text': row['text']})

    merged_subtitles = merge_ocr_texts(ocr_text_data)

    # SRT 파일 생성
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    # 자막 생성
    video_filename = os.path.splitext(os.path.basename(csv_filename))[0]
    with open(f'./uploads/{video_filename}.srt', 'w', encoding='utf-8') as f:
        for idx, subtitle in enumerate(merged_subtitles, start=1):
            start = format_time(subtitle['start_time'])
            end = format_time(subtitle['end_time'])
            subtitle_line = subtitle['text'].replace("\\n", "\n")
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{subtitle_line}\n\n")

if __name__ == "__main__":
    # 메인 Tk 창 생성
    root = Tk()
    root.withdraw()  # 메인 윈도우 숨기기

    # 파일 선택 대화 상자를 항상 맨 위로 설정
    root.attributes('-topmost', True)
    root.update()

    csv_filename = askopenfilename(parent=root, filetypes=[("CSV files", "*.csv")])
    test_merge_ocr_texts_from_csv(csv_filename)
