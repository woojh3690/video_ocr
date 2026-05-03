import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from core.jsonl_to_srt import jsonl_to_srt
from core.ocr_types import SpottingItem


def make_direct_crop_record(frame_number: int, text: str) -> dict:
    item = SpottingItem(
        text=text,
        quad=((0, 0), (1000, 0), (1000, 1000), (0, 1000)),
    )
    return {
        "frame_number": frame_number,
        "time": round(frame_number / 10, 3),
        "spotting_items": [item.to_dict()],
        "ocr_mode": "crop",
        "ocr_area": [10, 8, 20, 16],
    }


class JsonlToSrtTests(unittest.TestCase):
    def test_direct_crop_mode_tracks_contiguous_text_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "direct.jsonl"
            records = [
                make_direct_crop_record(frame_number, "hello world")
                for frame_number in range(1, 6)
            ] + [
                make_direct_crop_record(frame_number, "goodbye world")
                for frame_number in range(6, 11)
            ]
            with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
                for record in records:
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            jsonl_to_srt(jsonl_path)

            srt_files = list(Path(temp_dir).glob("direct.*.srt"))
            self.assertEqual(len(srt_files), 1)
            srt_text = srt_files[0].read_text(encoding="utf-8")
            self.assertIn("hello world", srt_text)
            self.assertIn("goodbye world", srt_text)


if __name__ == "__main__":
    unittest.main()
