import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from merging_module import merge_ocr_texts

def test_merge_ocr_texts_from_csv(csv_filename):
    ocr_text_data = []
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ocr_text_data.append({'time': float(row['time']), 'text': row['text']})

    merged_subtitles = merge_ocr_texts(ocr_text_data)

    # 결과 출력 또는 SRT 파일 생성 등 원하는 작업 수행
    for idx, subtitle in enumerate(merged_subtitles, start=1):
        print(f"{idx}")
        print(f"{subtitle['start_time']} --> {subtitle['end_time']}")
        print(f"{subtitle['text']}\n")

if __name__ == "__main__":
    # 메인 Tk 창 생성
    root = Tk()
    root.withdraw()  # 메인 윈도우 숨기기

    # 파일 선택 대화 상자를 항상 맨 위로 설정
    root.attributes('-topmost', True)
    root.update()

    csv_filename = askopenfilename(parent=root, filetypes=[("CSV files", "*.csv")])
    test_merge_ocr_texts_from_csv(csv_filename)
