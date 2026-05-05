import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from core.jsonl_to_srt import Segment, _merge_nearby_identical_segments, jsonl_to_srt
from core.ocr_types import SpottingItem, TEXT_STATUS_TRUNCATED


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


def make_full_screen_record(frame_number: int, items: list[SpottingItem]) -> dict:
    return {
        "frame_number": frame_number,
        "time": round(frame_number / 10, 3),
        "spotting_items": [item.to_dict() for item in items],
        "ocr_mode": "full_screen",
        "ocr_area": [0, 0, 100, 100],
    }


class JsonlToSrtTests(unittest.TestCase):
    def test_nearby_identical_segments_are_merged(self):
        segments = [
            Segment(index=0, start=0.0, end=1.0, text="same subtitle"),
            Segment(index=0, start=1.9, end=3.0, text="same subtitle"),
            Segment(index=0, start=4.2, end=5.0, text="same subtitle"),
            Segment(index=0, start=5.5, end=6.0, text="different subtitle"),
        ]

        merged_segments = _merge_nearby_identical_segments(segments)

        self.assertEqual(len(merged_segments), 3)
        self.assertEqual(merged_segments[0].text, "same subtitle")
        self.assertEqual(merged_segments[0].start, 0.0)
        self.assertEqual(merged_segments[0].end, 3.0)

    def test_nearby_segments_with_different_text_are_not_merged(self):
        segments = [
            Segment(index=0, start=0.0, end=1.0, text="first subtitle"),
            Segment(index=0, start=1.5, end=2.0, text="second subtitle"),
        ]

        merged_segments = _merge_nearby_identical_segments(segments)

        self.assertEqual(len(merged_segments), 2)

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

    def test_truncated_text_is_not_used_as_subtitle_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "full.jsonl"
            normal_item = SpottingItem(
                text="stable subtitle",
                quad=((100, 100), (300, 100), (300, 180), (100, 180)),
            )
            truncated_item = SpottingItem(
                text="",
                quad=((600, 100), (800, 100), (800, 180), (600, 180)),
                text_status=TEXT_STATUS_TRUNCATED,
            )
            with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
                for frame_number in range(1, 10):
                    record = make_full_screen_record(frame_number, [normal_item, truncated_item])
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            jsonl_to_srt(jsonl_path)

            srt_files = list(Path(temp_dir).glob("full.*.srt"))
            self.assertEqual(len(srt_files), 1)
            srt_text = srt_files[0].read_text(encoding="utf-8")
            self.assertIn("stable subtitle", srt_text)
            self.assertNotIn(TEXT_STATUS_TRUNCATED, srt_text)

    def test_truncated_only_track_does_not_create_subtitle_segment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "truncated_only.jsonl"
            truncated_item = SpottingItem(
                text="",
                quad=((0, 0), (1000, 0), (1000, 1000), (0, 1000)),
                text_status=TEXT_STATUS_TRUNCATED,
            )
            with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
                for frame_number in range(1, 10):
                    record = make_direct_crop_record(frame_number, "")
                    record["spotting_items"] = [truncated_item.to_dict()]
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            jsonl_to_srt(jsonl_path)

            srt_files = list(Path(temp_dir).glob("truncated_only.*.srt"))
            self.assertEqual(len(srt_files), 1)
            self.assertEqual(srt_files[0].read_text(encoding="utf-8"), "")

    def test_missing_text_status_defaults_to_ok(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "legacy.jsonl"
            record = make_direct_crop_record(1, "legacy text")
            del record["spotting_items"][0]["text_status"]
            with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
                for frame_number in range(1, 5):
                    record["frame_number"] = frame_number
                    record["time"] = round(frame_number / 10, 3)
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            jsonl_to_srt(jsonl_path)

            srt_files = list(Path(temp_dir).glob("legacy.*.srt"))
            self.assertEqual(len(srt_files), 1)
            self.assertIn("legacy text", srt_files[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
