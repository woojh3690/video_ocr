from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Iterable, Tuple

import cv2
import numpy as np
from openai import AsyncOpenAI, LengthFinishReasonError
from pydantic import ValidationError

from core.hunyuan_client import OcrProcessingError, SpottingItem

TEXT_BBOX_ONLY_PROMPT = """
Detect ordinary visible text-region bounding boxes in this image.
Return only a JSON array. Each item must be an object: {"label":"Text","bbox":"x0 y0 x1 y1"}.
Coordinates are normalized 0-1000. Do not OCR text. Do not describe images.
Do not return non-text regions. If no text exists, return [].
""".strip()

PADDLE_OCR_PROMPT = "OCR:"
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_TEXT_JSON_BBOX_RE = re.compile(
    r'"label"\s*:\s*"Text".{0,80}?"?bbox"?\s*:\s*(?P<bbox>"[^"]+"|\[[^\]]+\])',
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class ChandraTextBlock:
    normalized_bbox: Tuple[int, int, int, int]
    pixel_bbox: Tuple[int, int, int, int]

    @property
    def normalized_quad(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        x1, y1, x2, y2 = self.normalized_bbox
        return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


class ChandraLayoutParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text_bbox_values: list[str] = []
        self._div_layout_stack: list[bool] = []
        self._layout_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "div":
            return

        attr_map = {key.lower(): value for key, value in attrs}
        label = (attr_map.get("data-label") or "").strip()
        bbox = (attr_map.get("data-bbox") or "").strip()
        is_layout_block = bool(label)
        is_top_level_layout = is_layout_block and self._layout_depth == 0

        if is_top_level_layout and label == "Text" and bbox:
            self.text_bbox_values.append(bbox)

        self._div_layout_stack.append(is_layout_block)
        if is_layout_block:
            self._layout_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "div" or not self._div_layout_stack:
            return

        is_layout_block = self._div_layout_stack.pop()
        if is_layout_block:
            self._layout_depth = max(0, self._layout_depth - 1)


class OpenAIVisionClient:
    def __init__(self, base_url: str | None, model: str, api_key: str = "dummy_key"):
        normalized_base_url = (base_url or "").strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("LLM Base URL 설정이 필요합니다.")
        if not normalized_base_url.endswith("/v1"):
            normalized_base_url = f"{normalized_base_url}/v1"
        if not model or not model.strip():
            raise ValueError("LLM 모델 설정이 필요합니다.")

        self.client = AsyncOpenAI(base_url=normalized_base_url, api_key=api_key)
        self.model = model.strip()

    async def complete_image(self, image_bgr: np.ndarray, prompt: str, max_tokens: int) -> str:
        data_url = image_to_data_url(image_bgr)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.0,
            stream=False,
            max_tokens=max_tokens,
        )
        return extract_response_text(response.choices[0].message.content)


class ChandraDetectorClient(OpenAIVisionClient):
    async def detect(
        self,
        frame_idx: int,
        image_bgr: np.ndarray,
    ) -> tuple[int, list[ChandraTextBlock], str | None]:
        try:
            content = await self.complete_image(image_bgr, TEXT_BBOX_ONLY_PROMPT, max_tokens=512)
            content = clean_model_text(content)
            blocks = parse_chandra_text_blocks(content, image_bgr.shape[1], image_bgr.shape[0])
            return frame_idx, blocks, None
        except (ValidationError, LengthFinishReasonError) as exc:
            print(f"[Warn] 프레임 {frame_idx} Chandra bbox 검출 중 예외 발생: {exc!r}")
            return frame_idx, [], None
        except Exception as exc:
            error = OcrProcessingError(frame_idx, exc)
            print(f"[Error] {error}")
            raise error


class PaddleOCRRecognizerClient(OpenAIVisionClient):
    async def recognize(
        self,
        frame_idx: int,
        image_bgr: np.ndarray,
    ) -> str:
        try:
            content = await self.complete_image(image_bgr, PADDLE_OCR_PROMPT, max_tokens=1024)
            return clean_plain_ocr_text(content)
        except (ValidationError, LengthFinishReasonError) as exc:
            print(f"[Warn] 프레임 {frame_idx} Paddle OCR 중 예외 발생: {exc!r}")
            return ""
        except Exception as exc:
            error = OcrProcessingError(frame_idx, exc)
            print(f"[Error] {error}")
            raise error


def image_to_data_url(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("이미지를 PNG로 인코딩하지 못했습니다.")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def extract_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return ""


def clean_model_text(text: str) -> str:
    candidate = re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL)
    code_block = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if code_block:
        candidate = code_block.group(1)
    return candidate.strip()


def clean_plain_ocr_text(text: str) -> str:
    candidate = clean_model_text(text)
    candidate = re.sub(r"^OCR:\s*", "", candidate, flags=re.IGNORECASE)
    return candidate.strip()


def parse_chandra_text_blocks(content: str, image_width: int, image_height: int) -> list[ChandraTextBlock]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("이미지 크기가 올바르지 않습니다.")

    json_bbox_values = parse_chandra_json_bbox_values(content or "")
    if json_bbox_values is not None:
        blocks: list[ChandraTextBlock] = []
        for bbox_value in json_bbox_values:
            block = build_chandra_text_block(bbox_value, image_width, image_height)
            if block is not None:
                blocks.append(block)
        return dedup_blocks(blocks)

    parser = ChandraLayoutParser()
    parser.feed(content or "")

    blocks: list[ChandraTextBlock] = []
    for bbox_value in parser.text_bbox_values:
        parsed = parse_bbox_value(bbox_value)
        if parsed is None:
            continue
        block = build_chandra_text_block(parsed, image_width, image_height)
        if block is not None:
            blocks.append(block)
    return dedup_blocks(blocks)


def parse_bbox_value(value: str) -> tuple[float, float, float, float] | None:
    numbers = [float(item) for item in _NUMBER_RE.findall(value)]
    if len(numbers) < 4:
        return None
    return numbers[0], numbers[1], numbers[2], numbers[3]


def parse_chandra_json_bbox_values(content: str) -> list[tuple[float, float, float, float]] | None:
    candidate = clean_model_text(content)
    if not candidate or candidate[0] not in "[{":
        return None

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return parse_chandra_malformed_json_bbox_values(candidate)

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return None

    bboxes: list[tuple[float, float, float, float]] = []
    for item in payload:
        parsed = parse_chandra_json_bbox_item(item)
        if parsed is not None:
            bboxes.append(parsed)
    return bboxes


def parse_chandra_json_bbox_item(item: Any) -> tuple[float, float, float, float] | None:
    if isinstance(item, dict):
        label = item.get("label")
        if label is not None and str(label).strip() != "Text":
            return None
        bbox = item.get("bbox") or item.get("data-bbox") or item.get("box")
        return parse_json_bbox_value(bbox)

    if isinstance(item, (list, tuple)):
        return parse_json_bbox_value(item)

    return None


def parse_chandra_malformed_json_bbox_values(content: str) -> list[tuple[float, float, float, float]]:
    bboxes: list[tuple[float, float, float, float]] = []
    for match in _TEXT_JSON_BBOX_RE.finditer(content):
        bbox_value = match.group("bbox").strip().strip('"')
        parsed = parse_bbox_value(bbox_value)
        if parsed is not None:
            bboxes.append(parsed)
    return bboxes


def parse_json_bbox_value(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, str):
        return parse_bbox_value(value)

    if isinstance(value, (list, tuple)):
        numbers: list[float] = []
        for item in value:
            if isinstance(item, (int, float)):
                numbers.append(float(item))
                continue
            if isinstance(item, str):
                try:
                    numbers.append(float(item.strip()))
                except ValueError:
                    return None
                continue
            return None
        if len(numbers) < 4:
            return None
        return numbers[0], numbers[1], numbers[2], numbers[3]

    return None


def build_chandra_text_block(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> ChandraTextBlock | None:
    x1, y1, x2, y2 = bbox
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))

    coordinate_mode = infer_coordinate_mode((left, top, right, bottom), image_width, image_height)
    pixel_bbox = restore_bbox_to_pixels((left, top, right, bottom), image_width, image_height, coordinate_mode)
    px1, py1, px2, py2 = pixel_bbox
    if px2 <= px1 or py2 <= py1:
        return None

    if coordinate_mode == "normalized_1000":
        normalized_bbox = tuple(clamp_int(round(value), 0, 1000) for value in (left, top, right, bottom))
    else:
        normalized_bbox = normalize_pixel_bbox(pixel_bbox, image_width, image_height)

    return ChandraTextBlock(
        normalized_bbox=normalized_bbox,  # type: ignore[arg-type]
        pixel_bbox=pixel_bbox,
    )


def infer_coordinate_mode(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> str:
    max_value = max(abs(value) for value in bbox)
    if max_value <= 1.2:
        return "unit"
    if max_value <= 1000.0:
        return "normalized_1000"
    return "identity"


def restore_bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    coordinate_mode: str,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    if coordinate_mode == "unit":
        values = (
            x1 * (image_width - 1),
            y1 * (image_height - 1),
            x2 * (image_width - 1),
            y2 * (image_height - 1),
        )
    elif coordinate_mode == "normalized_1000":
        values = (
            (x1 / 1000.0) * (image_width - 1),
            (y1 / 1000.0) * (image_height - 1),
            (x2 / 1000.0) * (image_width - 1),
            (y2 / 1000.0) * (image_height - 1),
        )
    else:
        values = (x1, y1, x2, y2)

    px1, py1, px2, py2 = (int(round(value)) for value in values)
    return (
        clamp_int(px1, 0, image_width - 1),
        clamp_int(py1, 0, image_height - 1),
        clamp_int(px2, 0, image_width - 1),
        clamp_int(py2, 0, image_height - 1),
    )


def normalize_pixel_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        normalize_pixel_value(x1, image_width),
        normalize_pixel_value(y1, image_height),
        normalize_pixel_value(x2, image_width),
        normalize_pixel_value(y2, image_height),
    )


def normalize_pixel_value(value: int, size: int) -> int:
    if size <= 1:
        return 0
    return clamp_int(round((value / (size - 1)) * 1000.0), 0, 1000)


def crop_with_padding(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    image_height, image_width = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)
    if box_width <= 0 or box_height <= 0:
        return None

    padding = int(round(max(8.0, max(box_width, box_height) * 0.04)))
    crop_x1 = clamp_int(x1 - padding, 0, image_width)
    crop_y1 = clamp_int(y1 - padding, 0, image_height)
    crop_x2 = clamp_int(x2 + padding + 1, 0, image_width)
    crop_y2 = clamp_int(y2 + padding + 1, 0, image_height)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    return image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]


def spotting_item_from_block(block: ChandraTextBlock, text: str) -> SpottingItem | None:
    cleaned_text = clean_plain_ocr_text(text)
    if not cleaned_text:
        return None
    return SpottingItem(text=cleaned_text, quad=block.normalized_quad)


def dedup_blocks(blocks: Iterable[ChandraTextBlock]) -> list[ChandraTextBlock]:
    seen: set[tuple[int, int, int, int]] = set()
    deduped: list[ChandraTextBlock] = []
    for block in blocks:
        if block.normalized_bbox in seen:
            continue
        seen.add(block.normalized_bbox)
        deduped.append(block)
    return deduped


def clamp_int(value: int | float, minimum: int, maximum: int) -> int:
    return int(max(minimum, min(int(value), maximum)))
