from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple

from openai import AsyncOpenAI, LengthFinishReasonError
from pydantic import ValidationError


class OcrProcessingError(RuntimeError):
    def __init__(self, frame_number: int, original_exception: Exception):
        self.frame_number = frame_number
        self.original_exception = original_exception
        message = f"프레임 {frame_number} OCR 중 복구하지 못한 오류: {original_exception}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SpottingItem:
    text: str
    quad: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in self.quad]
        ys = [p[1] for p in self.quad]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def center(self) -> Tuple[float, float]:
        xs = [p[0] for p in self.quad]
        ys = [p[1] for p in self.quad]
        return (sum(xs) / 4.0, sum(ys) / 4.0)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["quad"] = [[int(x), int(y)] for (x, y) in self.quad]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SpottingItem":
        quad_raw = data["quad"]
        if not (isinstance(quad_raw, list) and len(quad_raw) == 4):
            raise ValueError(f"quad 형식이 이상함: {quad_raw!r}")

        quad = tuple((int(point[0]), int(point[1])) for point in quad_raw)  # type: ignore[arg-type]
        if len(quad) != 4:
            raise ValueError("quad 꼭짓점은 4개여야 합니다.")

        return cls(text=str(data["text"]), quad=quad)  # type: ignore[arg-type]


class HunyuanOCRClient:
    def __init__(self, base_url: str | None, model: str, api_key: str = "dummy_key"):

        normalized_base_url = (base_url or "").strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("llm_base_url 설정이 필요합니다.")
        if not normalized_base_url.endswith("/v1"):
            normalized_base_url = f"{normalized_base_url}/v1"
        normalized_base_url = normalized_base_url

        self.client = AsyncOpenAI(base_url=normalized_base_url, api_key=api_key)
        self.model = model
        self._loc_re = re.compile(r"<?\|LOC_(\d+(?:\.\d+)?)\|>", flags=re.IGNORECASE)
        self._coord_pair_re = re.compile(
            r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
        )
        self._spotting_line_re = re.compile(
            r"^\s*(?P<text>.+?)\s*(?P<coords>\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)\s*,\s*\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\))\s*$"
        )
        self._full_width_translation = str.maketrans({
            "\uff08": "(",
            "\uff09": ")",
            "\uff0c": ",",
        })

    async def predict(
        self,
        frame_idx: int,
        base64_img: str,
        image_width: int,
        image_height: int,
    ) -> tuple[int, List[SpottingItem]]:
        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                    },
                    {
                        "type": "text",
                        "text": "检测并识别图片中的文字，将文本坐标格式化输出。",
                    },
                ],
            },
        ]

        spotting_list: list[SpottingItem] = []
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                stream=False,
                max_tokens=4096,
            )
            content = self.extract_response_text(response.choices[0].message.content)
            content = self.clean_repeated_substrings(content)
            if content.strip():
                spotting_list = self.parse_to_spotting_items(content, image_width, image_height)
            spotting_list = self.dedup_spotting_items(spotting_list)
        except (ValidationError, LengthFinishReasonError) as exc:
            print(f"[Warn] 프레임 {frame_idx} OCR 중 예외 발생: {exc!r}")
            spotting_list = []
        except Exception as exc:
            error = OcrProcessingError(frame_idx, exc)
            print(f"[Error] {error}")
            raise error

        return frame_idx, spotting_list

    def parse_to_spotting_items(
        self,
        content: str,
        image_width: int,
        image_height: int,
    ) -> list[SpottingItem]:
        last_error: Optional[Exception] = None
        for candidate in self.iter_parser_candidates(content):
            try:
                predictions = self.parse_predictions(candidate)
                return self.build_spotting_items(predictions, image_width, image_height)
            except ValueError as exc:
                last_error = exc

        if last_error is not None:
            print(f"[Warn] HunyuanOCR 응답 파싱 실패: {last_error}")
            print("[Warn] HunyuanOCR 원본 LLM 출력:", content)
        return []

    def iter_parser_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []
        for candidate in (text, self.repair_parser_input(text)):
            cleaned = candidate.strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)
        return candidates

    def repair_parser_input(self, text: str) -> str:
        repaired = text.strip()
        repaired = repaired.replace("??, '", '"').replace("??, \"", '"').replace("??, ", '"')
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        if repaired.count("'") > repaired.count('"'):
            repaired = repaired.replace("'", '"')
        return repaired

    def extract_response_text(self, content: Any) -> str:
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

    def clean_repeated_substrings(self, text: str) -> str:
        length = len(text)
        if length < 8000:
            return text

        for repeat_length in range(2, length // 10 + 1):
            candidate = text[-repeat_length:]
            repeat_count = 0
            index = length - repeat_length

            while index >= 0 and text[index:index + repeat_length] == candidate:
                repeat_count += 1
                index -= repeat_length

            if repeat_count >= 10:
                return text[:length - repeat_length * (repeat_count - 1)]

        return text

    def parse_predictions(self, text: str) -> List[dict[str, Any]]:
        candidate = self.prepare_text(text)
        if not candidate:
            raise ValueError("Empty response")

        try:
            payload = self.load_json_payload(candidate)
            return self.parse_prediction_payload(payload)
        except ValueError:
            pass

        for parser in (
            self.parse_location_token_predictions,
            self.parse_tuple_spotting_output,
            self.parse_compact_xyxy_pairs,
            self.parse_line_predictions,
        ):
            try:
                return parser(candidate)
            except ValueError:
                continue

        raise ValueError("Unsupported spotting output format")

    def prepare_text(self, text: str) -> str:
        candidate = text.strip()
        candidate = re.sub(r"<think>.*?</think>\s*", "", candidate, flags=re.DOTALL)
        code_block = re.search(r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL | re.IGNORECASE)
        if code_block:
            candidate = code_block.group(1).strip()
        candidate = candidate.translate(self._full_width_translation)
        candidate = candidate.replace("\r\n", "\n").replace("\r", "\n")
        candidate = candidate.replace("```text", "").replace("```json", "").replace("```", "")
        candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
        return candidate.strip()

    def load_json_payload(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return json.loads(self.extract_first_json_payload(text))

    def extract_first_json_payload(self, text: str) -> str:
        starts = [idx for idx in (text.find("["), text.find("{")) if idx >= 0]
        if not starts:
            raise ValueError("No JSON payload found in response")

        start = min(starts)
        opening = text[start]
        closing = "]" if opening == "[" else "}"
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        raise ValueError("Unbalanced JSON payload in response")

    def parse_prediction_payload(self, payload: Any) -> List[dict[str, Any]]:
        if isinstance(payload, dict):
            for key in ("predictions", "annotations", "results", "items", "regions"):
                if key in payload:
                    payload = payload[key]
                    break
            else:
                raise ValueError("Prediction payload must be a list or contain predictions")
        if not isinstance(payload, list):
            raise ValueError("Prediction payload must be a list")

        predictions: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"Prediction {index} must be an object")

            text_value = (
                item.get("text")
                or item.get("text_content")
                or item.get("transcript")
                or item.get("label")
                or item.get("content")
            )
            if text_value is None:
                raise ValueError(f"Prediction {index} missing text")

            points, coordinate_mode = self.extract_prediction_points(item, index)
            prediction = {
                "text": str(text_value),
                "points": points,
            }
            if coordinate_mode is not None:
                prediction["coordinate_mode"] = coordinate_mode
            confidence = item.get("confidence")
            if confidence is not None:
                prediction["confidence"] = float(confidence)
            predictions.append(prediction)

        if not predictions:
            raise ValueError("No predictions found in payload")
        return predictions

    def extract_prediction_points(
        self,
        item: dict[str, Any],
        index: int,
    ) -> tuple[List[List[float]], Optional[str]]:
        context = f"prediction[{index}]"

        if "bbox_2d" in item:
            return self.normalize_bbox(item["bbox_2d"], context), "normalized_1000"

        for key in ("bbox", "box", "quad", "polygon", "poly", "points"):
            if key in item:
                return self.normalize_bbox(item[key], context), None

        if all(key in item for key in ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")):
            return self.normalize_points(
                [
                    [item["x1"], item["y1"]],
                    [item["x2"], item["y2"]],
                    [item["x3"], item["y3"]],
                    [item["x4"], item["y4"]],
                ],
                context,
            ), None

        raise ValueError(f"Prediction {index} missing bbox coordinates")

    def normalize_bbox(self, bbox: Any, context: str) -> List[List[float]]:
        if isinstance(bbox, dict):
            if "points" in bbox:
                return self.normalize_points(bbox["points"], context)
            if "xyxy" in bbox:
                return self.xyxy_to_points(bbox["xyxy"], context)
            if all(key in bbox for key in ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")):
                return self.normalize_points(
                    [
                        [bbox["x1"], bbox["y1"]],
                        [bbox["x2"], bbox["y2"]],
                        [bbox["x3"], bbox["y3"]],
                        [bbox["x4"], bbox["y4"]],
                    ],
                    context,
                )

        if not isinstance(bbox, list):
            raise ValueError(f"{context} bbox must be a list or dict")
        if len(bbox) == 4 and all(isinstance(value, (int, float)) for value in bbox):
            return self.xyxy_to_points(bbox, context)
        if len(bbox) == 4 and all(isinstance(value, list) and len(value) == 2 for value in bbox):
            return [[float(point[0]), float(point[1])] for point in bbox]
        raise ValueError(f"{context} bbox format is not supported")

    def xyxy_to_points(self, bbox: Any, context: str) -> List[List[float]]:
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{context} xyxy bbox must contain 4 values")
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ]

    def normalize_points(self, points: Any, context: str) -> List[List[float]]:
        if not isinstance(points, list) or len(points) != 4:
            raise ValueError(f"{context} must contain exactly 4 points")
        normalized = []
        for point_index, point in enumerate(points):
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"{context} point[{point_index}] must be [x, y]")
            normalized.append([float(point[0]), float(point[1])])
        return normalized

    def parse_location_token_predictions(self, text: str) -> List[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            loc_values = self._loc_re.findall(line)
            if len(loc_values) < 8:
                continue

            coords = [float(value) for value in loc_values[:8]]
            content = self._loc_re.sub("", line).strip().strip('"')
            if not content:
                continue

            predictions.append(
                {
                    "text": content,
                    "points": [
                        [coords[0], coords[1]],
                        [coords[2], coords[3]],
                        [coords[4], coords[5]],
                        [coords[6], coords[7]],
                    ],
                    "coordinate_mode": "normalized_1000",
                }
            )

        if predictions:
            return predictions
        raise ValueError("Unsupported location token output format")

    def parse_compact_xyxy_pairs(self, text: str) -> List[dict[str, Any]]:
        pattern = re.compile(
            r"(.*?)(?:\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\))",
            flags=re.DOTALL,
        )
        predictions: list[dict[str, Any]] = []
        for match in pattern.finditer(text):
            content = match.group(1).strip().strip('"')
            if not content:
                continue

            x1, y1, x2, y2 = [float(value) for value in match.groups()[1:5]]
            predictions.append(
                {
                    "text": content,
                    "points": self.xyxy_to_points([x1, y1, x2, y2], "compact_xyxy"),
                    "coordinate_mode": "normalized_1000",
                }
            )

        if predictions:
            return predictions
        raise ValueError("Unsupported compact xyxy output format")

    def parse_tuple_spotting_output(self, text: str) -> List[dict[str, Any]]:
        point = r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
        plain_point = r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)"
        next_box = rf"{plain_point}\s*,\s*{plain_point}\s*,\s*{plain_point}\s*,\s*{plain_point}"
        pattern = re.compile(
            rf"{point}\s*,\s*{point}\s*,\s*{point}\s*,\s*{point}\s*(.*?)(?={next_box}|$)",
            flags=re.DOTALL,
        )
        predictions: list[dict[str, Any]] = []
        for match in pattern.finditer(text):
            coords = [float(value) for value in match.groups()[:8]]
            content = match.group(9).strip().strip('"')
            if not content:
                continue

            predictions.append(
                {
                    "text": content,
                    "points": [
                        [coords[0], coords[1]],
                        [coords[2], coords[3]],
                        [coords[4], coords[5]],
                        [coords[6], coords[7]],
                    ],
                    "coordinate_mode": "normalized_1000",
                }
            )

        if predictions:
            return predictions
        raise ValueError("Unsupported tuple spotting output format")

    def parse_line_predictions(self, text: str) -> List[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        coord_matches = list(self._coord_pair_re.finditer(text))

        for line in text.split("\n"):
            item = self.parse_spotting_bbox_line(line)
            if item is not None:
                predictions.append(item)

        if predictions and len(predictions) == len(coord_matches):
            return predictions

        cursor = 0
        predictions = []
        for match in coord_matches:
            content = text[cursor:match.start()].split("\n")[-1].strip()
            item = self.build_prediction_item(content, match.groups(), "normalized_1000")
            if item is not None:
                predictions.append(item)
            cursor = match.end()

        if predictions:
            return predictions

        pattern = re.compile(
            r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*[:\-]?\s*([^\n]+)"
        )
        predictions = []
        for match in pattern.finditer(text):
            x1, y1, x2, y2 = [float(value) for value in match.groups()[:4]]
            content = match.group(5).strip().strip('"')
            if not content:
                continue

            predictions.append(
                {
                    "text": content,
                    "points": self.xyxy_to_points([x1, y1, x2, y2], "line_prediction"),
                    "coordinate_mode": "normalized_1000",
                }
            )

        if predictions:
            return predictions
        raise ValueError("Unsupported line prediction format")

    def parse_spotting_bbox_line(self, line: str) -> Optional[dict[str, Any]]:
        match = self._spotting_line_re.fullmatch(line)
        if match is None:
            return None

        coord_match = self._coord_pair_re.fullmatch(match.group("coords"))
        if coord_match is None:
            return None

        return self.build_prediction_item(match.group("text"), coord_match.groups(), "normalized_1000")

    def build_prediction_item(
        self,
        text: str,
        coords: tuple[str, str, str, str],
        coordinate_mode: Optional[str],
    ) -> Optional[dict[str, Any]]:
        cleaned_text = re.sub(r"</?(?:ref|box|quad)>", "", text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip().strip("`").strip('"')
        if not cleaned_text:
            return None

        x1, y1, x2, y2 = [float(value) for value in coords]
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        item = {
            "text": cleaned_text,
            "points": [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ],
        }
        if coordinate_mode is not None:
            item["coordinate_mode"] = coordinate_mode
        return item

    def build_spotting_items(
        self,
        predictions: List[dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> list[SpottingItem]:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Invalid image size")

        coordinate_mode = self.infer_coordinate_mode(predictions, image_width, image_height)
        spotting_items: list[SpottingItem] = []
        for prediction in predictions:
            restored_points = [
                self.restore_point(point, image_width, image_height, coordinate_mode)
                for point in prediction["points"]
            ]
            normalized_quad = tuple(
                self.normalize_pixel_point(point, image_width, image_height)
                for point in restored_points
            )
            spotting_items.append(
                SpottingItem(
                    text=str(prediction["text"]),
                    quad=normalized_quad,  # type: ignore[arg-type]
                )
            )
        return spotting_items

    def infer_coordinate_mode(
        self,
        predictions: List[dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> str:
        declared_modes = {
            str(item.get("coordinate_mode"))
            for item in predictions
            if item.get("coordinate_mode")
        }
        if len(declared_modes) == 1:
            return next(iter(declared_modes))

        max_x = max(float(point[0]) for item in predictions for point in item["points"])
        max_y = max(float(point[1]) for item in predictions for point in item["points"])

        if max_x <= 1.2 and max_y <= 1.2:
            return "unit"
        if image_width <= 1000 and image_height <= 1000:
            if max_x <= image_width * 1.05 and max_y <= image_height * 1.05:
                return "identity"
        if max_x <= 1000.0 and max_y <= 1000.0:
            return "normalized_1000"
        return "identity"

    def restore_point(
        self,
        point: List[float],
        image_width: int,
        image_height: int,
        coordinate_mode: str,
    ) -> tuple[int, int]:
        x_value = float(point[0])
        y_value = float(point[1])

        if coordinate_mode == "unit":
            x_value *= image_width
            y_value *= image_height
        elif coordinate_mode == "normalized_1000":
            x_value = (x_value / 1000.0) * image_width
            y_value = (y_value / 1000.0) * image_height

        x_value = max(0.0, min(round(x_value), float(image_width)))
        y_value = max(0.0, min(round(y_value), float(image_height)))
        return int(x_value), int(y_value)

    def normalize_pixel_point(
        self,
        point: tuple[int, int],
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        x_value, y_value = point

        if image_width <= 1:
            normalized_x = 0
        else:
            normalized_x = int(round((x_value / (image_width - 1)) * 1000.0))

        if image_height <= 1:
            normalized_y = 0
        else:
            normalized_y = int(round((y_value / (image_height - 1)) * 1000.0))

        normalized_x = max(0, min(normalized_x, 1000))
        normalized_y = max(0, min(normalized_y, 1000))
        return normalized_x, normalized_y

    def dedup_spotting_items(self, items: List[SpottingItem]) -> List[SpottingItem]:
        seen: set[Tuple[str, Tuple[Tuple[int, int], ...]]] = set()
        deduped: List[SpottingItem] = []

        for item in items:
            key = (item.text, item.quad)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        return deduped
