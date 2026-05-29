import unittest
import asyncio
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.split_ocr_client import (
    SuryaDetectorClient,
    TextBlock,
    VisionCompletionResult,
    clean_plain_ocr_text,
    crop_with_padding,
    filter_blank_text_blocks,
    parse_surya_text_blocks,
)


class SplitOcrClientTests(unittest.TestCase):
    def test_parse_surya_text_blocks_uses_only_top_level_text(self):
        html = """
        <div data-label="Text" data-bbox="100 200 300 400"><p>hello</p></div>
        <div data-label="Table" data-bbox="10 10 90 90">
            <div data-label="Text" data-bbox="1 1 2 2">nested</div>
        </div>
        <div data-label="Image" data-bbox="500 500 900 900"></div>
        """

        blocks = parse_surya_text_blocks(html, 2000, 1000)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].normalized_bbox, (100, 200, 300, 400))

    def test_normalized_bbox_restores_to_pixel_bbox(self):
        html = '<div data-label="Text" data-bbox="100 250 500 750">text</div>'

        blocks = parse_surya_text_blocks(html, 1001, 501)

        self.assertEqual(blocks[0].pixel_bbox, (100, 125, 500, 375))

    def test_json_bbox_response_ignores_visual_layout_items(self):
        content = """
        [
          {"label": "Image", "bbox": "0 0 1000 1000"},
          {"label": "Figure", "bbox": "10 10 900 900"},
          {"label": "Text", "bbox": "34 136 122 707"}
        ]
        """

        blocks = parse_surya_text_blocks(content, 1920, 1080)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].normalized_bbox, (34, 136, 122, 707))

    def test_json_bbox_response_keeps_official_ocr_candidate_labels(self):
        content = """
        [
          {"label": "Table", "bbox": "10 20 300 400", "count": 50},
          {"label": "List-Group", "bbox": "310 20 500 400", "count": 50},
          {"label": "Code-Block", "bbox": "510 20 700 400", "count": 50},
          {"label": "Equation-Block", "bbox": "710 20 900 400", "count": 50},
          {"label": "Blank-Page", "bbox": "0 0 1000 1000", "count": 0}
        ]
        """

        blocks = parse_surya_text_blocks(content, 1920, 1080)

        self.assertEqual(
            [block.normalized_bbox for block in blocks],
            [
                (10, 20, 300, 400),
                (310, 20, 500, 400),
                (510, 20, 700, 400),
                (710, 20, 900, 400),
            ],
        )

    def test_full_frame_text_bbox_is_ignored(self):
        content = """
        [
          {"label": "Text", "bbox": "0 0 1000 1000"},
          {"label": "Text", "bbox": "3 2 998 999"},
          {"label": "Text", "bbox": "34 136 122 707"}
        ]
        """

        blocks = parse_surya_text_blocks(content, 1920, 1080)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].normalized_bbox, (34, 136, 122, 707))

    def test_blank_uniform_text_bbox_is_ignored(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[45:65, 15:25] = 255
        blocks = [
            TextBlock(normalized_bbox=(0, 0, 1000, 100), pixel_bbox=(0, 0, 99, 9)),
            TextBlock(normalized_bbox=(100, 400, 350, 700), pixel_bbox=(10, 40, 35, 70)),
        ]

        filtered = filter_blank_text_blocks(blocks, image)

        self.assertEqual([block.normalized_bbox for block in filtered], [(100, 400, 350, 700)])

    def test_malformed_json_bbox_response_is_recovered(self):
        content = '[{"label": "Text",bbox": "41 78 132 680"}, {"label": "Text",bbox": "299 277 494 305"}]'

        blocks = parse_surya_text_blocks(content, 1920, 1080)

        self.assertEqual([block.normalized_bbox for block in blocks], [(41, 78, 132, 680), (299, 277, 494, 305)])

    def test_crop_with_padding_is_clamped_to_image_bounds(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        crop = crop_with_padding(image, (0, 0, 20, 10))

        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[:2], (19, 29))

    def test_clean_plain_ocr_text_handles_empty_and_list_joined_text(self):
        self.assertEqual(clean_plain_ocr_text(" OCR:  hello \n"), "hello")
        self.assertEqual(clean_plain_ocr_text("   "), "")

    def test_empty_surya_response_has_no_blocks(self):
        self.assertEqual(parse_surya_text_blocks("", 1920, 1080), [])

    def test_surya_detector_limits_bbox_generation_tokens(self):
        class RecordingDetector(SuryaDetectorClient):
            def __init__(self):
                self.max_tokens = None
                self.prompt = None
                self.extra_body = None

            async def complete_image(self, image_bgr, prompt, max_tokens=None, extra_body=None):
                self.max_tokens = max_tokens
                self.prompt = prompt
                self.extra_body = extra_body
                return VisionCompletionResult(text="[]")

        detector = RecordingDetector()

        frame_number, blocks, _ = asyncio.run(detector.detect(1, np.zeros((8, 8, 3), dtype=np.uint8)))

        self.assertEqual(frame_number, 1)
        self.assertEqual(blocks, [])
        self.assertEqual(detector.max_tokens, 512)
        self.assertIn("Output the layout of this image as JSON", detector.prompt)
        self.assertIn('"label", "bbox", and "count"', detector.prompt)
        self.assertNotIn("HTML", detector.prompt)
        self.assertIn("structured_outputs", detector.extra_body)


if __name__ == "__main__":
    unittest.main()
