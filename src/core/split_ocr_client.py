import base64
import json
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Iterable, Tuple

import cv2
import numpy as np
from openai import AsyncOpenAI, LengthFinishReasonError
from pydantic import ValidationError

from core.ocr_types import OcrProcessingError, SpottingItem, TEXT_STATUS_OK, TEXT_STATUS_TRUNCATED
from core.logging_utils import log_ocr_event
from core.util import clean_ocr_text


SURYA_HIGH_ACCURACY_BBOX_PROMPT = (
    "Output the layout of this image as JSON. Each entry is a dict with "
    '"label", "bbox", and "count" fields. Bbox is x0 y0 x1 y1, normalized 0-1000.'
)
# 기존 호출부와 테스트가 같은 프롬프트 상수를 재사용할 수 있도록 별칭을 유지합니다.
TEXT_BBOX_ONLY_PROMPT = SURYA_HIGH_ACCURACY_BBOX_PROMPT
FULL_FRAME_BBOX_MARGIN = 5
BLANK_WHITE_THRESHOLD = 245
BLANK_PIXEL_FRACTION = 0.99
UNIFORM_COLOR_STD = 8.0

SURYA_LAYOUT_LABEL_SET = [
    "Caption",
    "Footnote",
    "Equation-Block",
    "List-Group",
    "Page-Header",
    "Page-Footer",
    "Image",
    "Section-Header",
    "Table",
    "Text",
    "Complex-Block",
    "Code-Block",
    "Form",
    "Table-Of-Contents",
    "Figure",
    "Chemical-Block",
    "Diagram",
    "Bibliography",
    "Blank-Page",
]
SURYA_LAYOUT_JSON_SCHEMA = {
    "type": "array",
    "maxItems": 200,
    "items": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": SURYA_LAYOUT_LABEL_SET},
            "bbox": {"type": "string", "pattern": r"^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$"},
            "count": {"type": "integer", "minimum": 0, "maximum": 10000},
        },
        "required": ["label", "bbox", "count"],
        "additionalProperties": False,
    },
}

PADDLE_OCR_PROMPT = "OCR:"
SURYA_TEXT_LAYOUT_LABELS = {
    "Bibliography",
    "Caption",
    "Code",
    "Footnote",
    "PageFooter",
    "PageHeader",
    "SectionHeader",
    "Text",
}
SURYA_TEXT_LAYOUT_LABEL_KEYS = {re.sub(r"[\s_-]+", "", label).lower() for label in SURYA_TEXT_LAYOUT_LABELS}
SURYA_OCR_CANDIDATE_LAYOUT_LABELS = SURYA_TEXT_LAYOUT_LABELS | {
    "ChemicalBlock",
    "Code",
    "ComplexBlock",
    "Equation",
    "Form",
    "ListGroup",
    "Table",
    "TableOfContents",
}
SURYA_OCR_CANDIDATE_LAYOUT_LABEL_KEYS = {
    re.sub(r"[\s_-]+", "", label).lower() for label in SURYA_OCR_CANDIDATE_LAYOUT_LABELS
}
SURYA_LAYOUT_LABEL_ALIASES = {
    "Blank-Page": "BlankPage",
    "Chemical-Block": "ChemicalBlock",
    "Code-Block": "Code",
    "Complex-Block": "ComplexBlock",
    "Equation-Block": "Equation",
    "Image": "Picture",
    "List-Group": "ListGroup",
    "Page-Footer": "PageFooter",
    "Page-Header": "PageHeader",
    "Section-Header": "SectionHeader",
    "Table-Of-Contents": "TableOfContents",
}
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_TEXT_JSON_BBOX_RE = re.compile(
    r'"label"\s*:\s*"(?P<label>[^"]+)".{0,80}?"?bbox"?\s*:\s*(?P<bbox>"[^"]+"|\[[^\]]+\])',
    re.DOTALL,
)


def canonicalize_surya_label(label: str) -> str:
    cleaned_label = label.strip()
    return SURYA_LAYOUT_LABEL_ALIASES.get(cleaned_label, cleaned_label)


def is_surya_text_layout_label(label: str) -> bool:
    # 후속 PaddleOCR-VL crop 인식에 맞도록 Surya의 텍스트성 레이아웃만 통과시킵니다.
    canonical_label = canonicalize_surya_label(label)
    normalized_key = re.sub(r"[\s_-]+", "", canonical_label).lower()
    return bool(canonical_label) and normalized_key in SURYA_TEXT_LAYOUT_LABEL_KEYS


def is_surya_ocr_candidate_layout_label(label: str) -> bool:
    # 공식 layout JSON에서는 표와 수식처럼 별도 label을 가진 블록도 OCR 후보로 유지합니다.
    canonical_label = canonicalize_surya_label(label)
    normalized_key = re.sub(r"[\s_-]+", "", canonical_label).lower()
    return bool(canonical_label) and normalized_key in SURYA_OCR_CANDIDATE_LAYOUT_LABEL_KEYS


@dataclass(frozen=True, slots=True)
class VisionCompletionResult:
    # LLM 응답 본문과 종료 사유를 함께 전달합니다.
    text: str
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RecognizedText:
    # recognizer 결과가 자막 후보로 쓸 수 있는지 상태를 보존합니다.
    text: str
    text_status: str = TEXT_STATUS_OK


@dataclass(frozen=True, slots=True)
class TextBlock:
    normalized_bbox: Tuple[int, int, int, int]
    pixel_bbox: Tuple[int, int, int, int]

    @property
    def normalized_quad(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        x1, y1, x2, y2 = self.normalized_bbox
        return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


class SuryaLayoutParser(HTMLParser):
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

        if is_top_level_layout and is_surya_text_layout_label(label) and bbox:
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
    def __init__(
        self,
        base_url: str | None,
        model: str,
        api_key: str = "dummy_key",
        log_component: str = "vllm-request",
        task_id: str | None = None,
    ):
        normalized_base_url = (base_url or "").strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("LLM Base URL 설정이 필요합니다.")
        if not normalized_base_url.endswith("/v1"):
            normalized_base_url = f"{normalized_base_url}/v1"
        if not model or not model.strip():
            raise ValueError("LLM 모델 설정이 필요합니다.")

        self.client = AsyncOpenAI(base_url=normalized_base_url, api_key=api_key)
        self.model = model.strip()
        self.log_component = log_component
        self.task_id = task_id
        self.submitted_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0

    async def complete_image(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        max_tokens: int | None = None,
        extra_body=None,
    ) -> VisionCompletionResult:
        data_url = image_to_data_url(image_bgr)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "stream": False,
            "extra_body": extra_body or {},
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        self.submitted_requests += 1
        request_started_at = time.monotonic()
        if self.submitted_requests == 1 or self.submitted_requests % 100 == 0:
            log_ocr_event(
                self.log_component,
                (
                    f"HTTP 요청 시작: count={self.submitted_requests}, model={self.model}, "
                    f"image={image_bgr.shape[1]}x{image_bgr.shape[0]}"
                ),
                self.task_id,
            )
        try:
            response = await self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            # 동시 실패 시 로그 폭주를 막으면서 최초 3건과 이후 10건 단위 오류를 남깁니다.
            self.failed_requests += 1
            if self.failed_requests <= 3 or self.failed_requests % 10 == 0:
                log_ocr_event(
                    self.log_component,
                    (
                        f"HTTP 요청 실패: count={self.submitted_requests}, failures={self.failed_requests}, "
                        f"elapsed={time.monotonic() - request_started_at:.2f}초, error={exc}"
                    ),
                    self.task_id,
                )
            raise
        self.completed_requests += 1
        if self.completed_requests == 1:
            log_ocr_event(
                self.log_component,
                f"첫 HTTP 응답 완료: elapsed={time.monotonic() - request_started_at:.2f}초",
                self.task_id,
            )
        choice = response.choices[0]
        return VisionCompletionResult(
            text=extract_response_text(choice.message.content),
            finish_reason=getattr(choice, "finish_reason", None),
        )


class SuryaDetectorClient(OpenAIVisionClient):
    async def detect(
        self,
        frame_idx: int,
        image_bgr: np.ndarray,
    ) -> tuple[int, list[TextBlock], str | None]:
        try:
            result = await self.complete_image(
                image_bgr,
                SURYA_HIGH_ACCURACY_BBOX_PROMPT,
                max_tokens=512,
                extra_body={"structured_outputs": {"json": SURYA_LAYOUT_JSON_SCHEMA}},
            )
            content = clean_model_text(result.text)
            blocks = parse_surya_text_blocks(content, image_bgr.shape[1], image_bgr.shape[0])
            filtered_blocks = filter_blank_text_blocks(blocks, image_bgr)
            if result.finish_reason == "length":
                if filtered_blocks:
                    print(
                        f"[Warn] 프레임 {frame_idx} Surya bbox 결과가 max_tokens로 중단되었지만 "
                        f"bbox {len(filtered_blocks)}개를 복구했습니다."
                    )
                    return frame_idx, filtered_blocks, None
                print(f"[Warn] 프레임 {frame_idx} Surya bbox 결과가 max_tokens로 중단되었습니다.")
                return frame_idx, [], None
            return frame_idx, filtered_blocks, None
        except (ValidationError, LengthFinishReasonError) as exc:
            print(f"[Warn] 프레임 {frame_idx} Surya bbox 검출 중 예외 발생: {exc!r}")
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
    ) -> RecognizedText:
        try:
            extra_body={ "repetition_penalty": 1.03 }
            result = await self.complete_image(image_bgr, PADDLE_OCR_PROMPT, max_tokens=256, extra_body=extra_body)
            if result.finish_reason == "length":
                print(f"[Warn] 프레임 {frame_idx} Paddle OCR 결과가 max_tokens로 중단되었습니다.")
                return RecognizedText(text="", text_status=TEXT_STATUS_TRUNCATED)
            return RecognizedText(text=clean_plain_ocr_text(result.text))
        except LengthFinishReasonError as exc:
            print(f"[Warn] 프레임 {frame_idx} Paddle OCR 결과가 max_tokens로 중단되었습니다: {exc!r}")
            return RecognizedText(text="", text_status=TEXT_STATUS_TRUNCATED)
        except ValidationError as exc:
            print(f"[Warn] 프레임 {frame_idx} Paddle OCR 중 예외 발생: {exc!r}")
            return RecognizedText(text="")
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
    # OCR 결과의 줄바꿈과 연속 공백을 하나의 공백으로 정규화합니다.
    return clean_ocr_text(candidate)


def parse_surya_text_blocks(content: str, image_width: int, image_height: int) -> list[TextBlock]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("이미지 크기가 올바르지 않습니다.")

    json_bbox_values = parse_layout_json_bbox_values(content or "")
    if json_bbox_values is not None:
        blocks: list[TextBlock] = []
        for bbox_value in json_bbox_values:
            block = build_text_block(bbox_value, image_width, image_height)
            if block is not None:
                blocks.append(block)
        return dedup_blocks(blocks)

    parser = SuryaLayoutParser()
    parser.feed(content or "")

    blocks: list[TextBlock] = []
    for bbox_value in parser.text_bbox_values:
        parsed = parse_bbox_value(bbox_value)
        if parsed is None:
            continue
        block = build_text_block(parsed, image_width, image_height)
        if block is not None:
            blocks.append(block)
    return dedup_blocks(blocks)


def parse_bbox_value(value: str) -> tuple[float, float, float, float] | None:
    numbers = [float(item) for item in _NUMBER_RE.findall(value)]
    if len(numbers) < 4:
        return None
    return numbers[0], numbers[1], numbers[2], numbers[3]


def parse_layout_json_bbox_values(content: str) -> list[tuple[float, float, float, float]] | None:
    candidate = clean_model_text(content)
    if not candidate or candidate[0] not in "[{":
        return None

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return parse_layout_malformed_json_bbox_values(candidate)

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return None

    bboxes: list[tuple[float, float, float, float]] = []
    for item in payload:
        parsed = parse_layout_json_bbox_item(item)
        if parsed is not None:
            bboxes.append(parsed)
    return bboxes


def parse_layout_json_bbox_item(item: Any) -> tuple[float, float, float, float] | None:
    if isinstance(item, dict):
        label = item.get("label")
        if label is not None and not is_surya_ocr_candidate_layout_label(str(label)):
            return None
        bbox = item.get("bbox") or item.get("data-bbox") or item.get("box")
        return parse_json_bbox_value(bbox)

    if isinstance(item, (list, tuple)):
        return parse_json_bbox_value(item)

    return None


def parse_layout_malformed_json_bbox_values(content: str) -> list[tuple[float, float, float, float]]:
    bboxes: list[tuple[float, float, float, float]] = []
    for match in _TEXT_JSON_BBOX_RE.finditer(content):
        label = match.group("label")
        if not is_surya_ocr_candidate_layout_label(label):
            continue
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


def build_text_block(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> TextBlock | None:
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

    return TextBlock(
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


def normalize_recognized_text(text: str | RecognizedText) -> RecognizedText:
    # 기존 테스트/호출부의 문자열 반환도 동일한 경로로 정규화합니다.
    if isinstance(text, RecognizedText):
        return text
    return RecognizedText(text=clean_plain_ocr_text(text))


def spotting_item_from_block(block: TextBlock, text: str | RecognizedText) -> SpottingItem | None:
    recognized_text = normalize_recognized_text(text)
    if recognized_text.text_status != TEXT_STATUS_OK:
        # 잘린 OCR은 텍스트를 버리되 bbox 추적용 항목은 유지합니다.
        return SpottingItem(
            text="",
            quad=block.normalized_quad,
            text_status=recognized_text.text_status,
        )

    cleaned_text = clean_plain_ocr_text(recognized_text.text)
    if not cleaned_text:
        return None
    return SpottingItem(text=cleaned_text, quad=block.normalized_quad)


def dedup_blocks(blocks: Iterable[TextBlock]) -> list[TextBlock]:
    seen: set[tuple[int, int, int, int]] = set()
    deduped: list[TextBlock] = []
    for block in blocks:
        if is_full_frame_text_block(block):
            continue
        if block.normalized_bbox in seen:
            continue
        seen.add(block.normalized_bbox)
        deduped.append(block)
    return deduped


def is_full_frame_text_block(block: TextBlock) -> bool:
    # Surya가 가끔 전체 화면을 Text bbox로 반환하는 오탐만 좁게 제거합니다.
    x1, y1, x2, y2 = block.normalized_bbox
    return (
        x1 <= FULL_FRAME_BBOX_MARGIN
        and y1 <= FULL_FRAME_BBOX_MARGIN
        and x2 >= 1000 - FULL_FRAME_BBOX_MARGIN
        and y2 >= 1000 - FULL_FRAME_BBOX_MARGIN
    )


def filter_blank_text_blocks(blocks: Iterable[TextBlock], image_bgr: np.ndarray) -> list[TextBlock]:
    # Surya 공식 LayoutPredictor처럼 빈 영역 또는 단색 영역 위의 Text 오탐을 제거합니다.
    filtered: list[TextBlock] = []
    for block in blocks:
        crop = crop_text_block_region(image_bgr, block)
        if crop is not None and is_blank_or_uniform_region(crop):
            continue
        filtered.append(block)
    return filtered


def crop_text_block_region(image_bgr: np.ndarray, block: TextBlock) -> np.ndarray | None:
    image_height, image_width = image_bgr.shape[:2]
    x1, y1, x2, y2 = block.pixel_bbox
    crop_x1 = clamp_int(x1, 0, image_width - 1)
    crop_y1 = clamp_int(y1, 0, image_height - 1)
    crop_x2 = clamp_int(x2 + 1, 0, image_width)
    crop_y2 = clamp_int(y2 + 1, 0, image_height)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    return image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]


def is_blank_or_uniform_region(image_bgr: np.ndarray) -> bool:
    # 흰 배경 문서와 검은 화면처럼 거의 단색인 영상 프레임을 모두 빈 영역으로 취급합니다.
    if image_bgr.size == 0:
        return False
    near_white_pixels = np.all(image_bgr >= BLANK_WHITE_THRESHOLD, axis=-1).mean()
    if float(near_white_pixels) > BLANK_PIXEL_FRACTION:
        return True
    pixels = image_bgr.reshape(-1, image_bgr.shape[-1])
    return float(pixels.std(axis=0).max()) < UNIFORM_COLOR_STD


def clamp_int(value: int | float, minimum: int, maximum: int) -> int:
    return int(max(minimum, min(int(value), maximum)))
