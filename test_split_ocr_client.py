import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from core.split_ocr_client import (
    clean_plain_ocr_text,
    crop_with_padding,
    parse_chandra_text_blocks,
)


class SplitOcrClientTests(unittest.TestCase):
    def test_parse_chandra_text_blocks_uses_only_top_level_text(self):
        html = """
        <div data-label="Text" data-bbox="100 200 300 400"><p>hello</p></div>
        <div data-label="Table" data-bbox="10 10 90 90">
            <div data-label="Text" data-bbox="1 1 2 2">nested</div>
        </div>
        <div data-label="Image" data-bbox="500 500 900 900"></div>
        """

        blocks = parse_chandra_text_blocks(html, 2000, 1000)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].normalized_bbox, (100, 200, 300, 400))

    def test_normalized_bbox_restores_to_pixel_bbox(self):
        html = '<div data-label="Text" data-bbox="100 250 500 750">text</div>'

        blocks = parse_chandra_text_blocks(html, 1001, 501)

        self.assertEqual(blocks[0].pixel_bbox, (100, 125, 500, 375))

    def test_json_bbox_response_uses_only_text_items(self):
        content = """
        [
          {"label": "Image", "bbox": "0 0 1000 1000"},
          {"label": "Text", "bbox": "34 136 122 707"}
        ]
        """

        blocks = parse_chandra_text_blocks(content, 1920, 1080)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].normalized_bbox, (34, 136, 122, 707))

    def test_malformed_json_bbox_response_is_recovered(self):
        content = '[{"label": "Text",bbox": "41 78 132 680"}, {"label": "Text",bbox": "299 277 494 305"}]'

        blocks = parse_chandra_text_blocks(content, 1920, 1080)

        self.assertEqual([block.normalized_bbox for block in blocks], [(41, 78, 132, 680), (299, 277, 494, 305)])

    def test_crop_with_padding_is_clamped_to_image_bounds(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        crop = crop_with_padding(image, (0, 0, 20, 10))

        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[:2], (19, 29))

    def test_clean_plain_ocr_text_handles_empty_and_list_joined_text(self):
        self.assertEqual(clean_plain_ocr_text(" OCR:  hello \n"), "hello")
        self.assertEqual(clean_plain_ocr_text("   "), "")

    def test_empty_chandra_response_has_no_blocks(self):
        self.assertEqual(parse_chandra_text_blocks("", 1920, 1080), [])


if __name__ == "__main__":
    unittest.main()
