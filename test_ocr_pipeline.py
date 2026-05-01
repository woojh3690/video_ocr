import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import core.ocr as ocr
from core.split_ocr_client import ChandraTextBlock


class OcrPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        self.stem = f"codex_ocr_pipeline_test_{id(self)}"
        self.video_name = f"{self.stem}.mp4"
        self.video_path = self.upload_dir / self.video_name
        self.jsonl_path = self.upload_dir / f"{self.stem}.jsonl"
        self.srt_path = self.upload_dir / f"{self.stem}.srt"
        self._write_video(frame_count=6)

    def tearDown(self):
        for path in (self.video_path, self.jsonl_path, self.srt_path):
            if path.exists():
                path.unlink()

    def _write_video(self, frame_count: int):
        writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (48, 48))
        for frame_index in range(frame_count):
            writer.write(np.full((48, 48, 3), frame_index * 20, dtype=np.uint8))
        writer.release()

    def _write_detector_cache(self, frame_numbers: range, with_blocks: bool = False):
        with self.jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for frame_number in frame_numbers:
                blocks = []
                if with_blocks:
                    blocks = [
                        {
                            "normalized_bbox": [0, 0, 500, 500],
                            "pixel_bbox": [0, 0, 24, 24],
                        }
                    ]
                jsonl_file.write(json.dumps({
                    "record_type": "detector",
                    "frame_number": frame_number,
                    "time": round(frame_number / 10, 3),
                    "detector_blocks": blocks,
                }) + "\n")

    def _write_ocr_cache(self, frame_numbers: range):
        with self.jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for frame_number in frame_numbers:
                jsonl_file.write(json.dumps({
                    "record_type": "ocr",
                    "frame_number": frame_number,
                    "time": round(frame_number / 10, 3),
                    "spotting_items": [],
                }) + "\n")

    async def _run_pipeline(self, detector_cls, recognizer_cls):
        with (
            patch.object(ocr, "ChandraDetectorClient", detector_cls),
            patch.object(ocr, "PaddleOCRRecognizerClient", recognizer_cls),
            patch.object(ocr, "jsonl_to_srt", lambda path: None),
        ):
            progress_values = []
            async for progress in ocr.process_ocr(
                self.video_name,
                0,
                0,
                48,
                48,
                full_screen_ocr=True,
                switch_to_recognizer=lambda: asyncio.sleep(0, result=True),
            ):
                progress_values.append(progress)
            return progress_values

    async def test_complete_detector_cache_skips_detector(self):
        self._write_detector_cache(range(1, 7))

        class DetectorMustNotRun:
            def __init__(self, *args, **kwargs):
                raise AssertionError("detector should be skipped")

        class FakeRecognizer:
            def __init__(self, *args, **kwargs):
                pass

            async def recognize(self, frame_idx, crop):
                return ""

        progress_values = await self._run_pipeline(DetectorMustNotRun, FakeRecognizer)

        self.assertEqual(progress_values[0], 50)
        self.assertEqual(progress_values[-1], 100)
        final_records = [
            json.loads(line)
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertTrue(final_records)
        self.assertTrue(all("record_type" not in record for record in final_records))

    async def test_incomplete_detector_cache_runs_detector(self):
        self._write_detector_cache(range(1, 4))
        calls = []

        class FakeDetector:
            def __init__(self, *args, **kwargs):
                pass

            async def detect(self, frame_idx, frame):
                calls.append(frame_idx)
                return frame_idx, [], None

        class FakeRecognizer:
            def __init__(self, *args, **kwargs):
                pass

            async def recognize(self, frame_idx, crop):
                return ""

        await self._run_pipeline(FakeDetector, FakeRecognizer)

        self.assertTrue(calls)
        self.assertEqual(calls, [4, 5, 6])

    async def test_recognizer_concurrency_keeps_jsonl_frame_order(self):
        self._write_detector_cache(range(1, 7), with_blocks=True)

        class DetectorMustNotRun:
            def __init__(self, *args, **kwargs):
                raise AssertionError("detector should be skipped")

        class SlowRecognizer:
            def __init__(self, *args, **kwargs):
                pass

            async def recognize(self, frame_idx, crop):
                await asyncio.sleep(0.03 if frame_idx % 2 else 0.01)
                return f"text-{frame_idx}"

        await self._run_pipeline(DetectorMustNotRun, SlowRecognizer)

        frame_numbers = [
            json.loads(line)["frame_number"]
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(frame_numbers, sorted(frame_numbers))

    async def test_existing_ocr_cache_skips_recognition_and_compacts(self):
        self._write_ocr_cache(range(1, 7))
        recognize_calls = []

        class DetectorMustNotRun:
            def __init__(self, *args, **kwargs):
                raise AssertionError("detector should be skipped")

        class FakeRecognizer:
            def __init__(self, *args, **kwargs):
                pass

            async def recognize(self, frame_idx, crop):
                recognize_calls.append(frame_idx)
                return ""

        await self._run_pipeline(DetectorMustNotRun, FakeRecognizer)

        self.assertEqual(recognize_calls, [])
        final_records = [
            json.loads(line)
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([record["frame_number"] for record in final_records], list(range(1, 7)))
        self.assertTrue(all("record_type" not in record for record in final_records))


if __name__ == "__main__":
    unittest.main()
