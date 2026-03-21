import re

from dataclasses import dataclass, asdict
from typing import Any, List, Optional, Tuple
from pydantic import ValidationError

from openai import AsyncOpenAI, LengthFinishReasonError

# llm 서버에서 치명적인 오류가 발생한 경우
class OcrProcessingError(RuntimeError):
    def __init__(self, frame_number: int, original_exception: Exception):
        self.frame_number = frame_number
        self.original_exception = original_exception
        message = f"프레임 {frame_number} OCR 중 예기치 못한 오류: {original_exception}"
        super().__init__(message)

@dataclass(frozen=True, slots=True)
class SpottingItem:
    text: str
    # 4개의 꼭짓점 (x1,y1)..(x4,y4)
    quad: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """(xmin, ymin, xmax, ymax)"""
        xs = [p[0] for p in self.quad]
        ys = [p[1] for p in self.quad]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def center(self) -> Tuple[float, float]:
        xs = [p[0] for p in self.quad]
        ys = [p[1] for p in self.quad]
        return (sum(xs) / 4.0, sum(ys) / 4.0)
    
    # ---- JSON 직렬화/역직렬화 헬퍼 ----
    def to_dict(self) -> dict:
        d = asdict(self)  # quad가 ( (x,y),... ) -> [[x,y], ...] 형태로 들어가게 됨(내부는 그대로일 수 있어도 json dump 시 list가 됨)
        # 명시적으로 JSON-friendly로 강제
        d["quad"] = [[int(x), int(y)] for (x, y) in self.quad]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SpottingItem":
        quad_raw = d["quad"]
        if not (isinstance(quad_raw, list) and len(quad_raw) == 4):
            raise ValueError(f"quad 형식이 이상함: {quad_raw!r}")

        quad = tuple((int(p[0]), int(p[1])) for p in quad_raw)  # type: ignore
        if len(quad) != 4:
            raise ValueError("quad 꼭짓점은 4개여야 합니다.")

        return cls(text=str(d["text"]), quad=quad)  # type: ignore

class PaddleClient:
    def __init__(self, base_url, model, api_key="dummy_key"):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        self._tasks = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
            "spotting": "检测并识别图片中的文字，将文本坐标格式化输出。",
            "seal": "Seal Recognition:",
        }
        self._loc_re = re.compile(r"<\|LOC_(\d+)\|>")
        self._coord_pair_re = re.compile(
            r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
        )
        self._spotting_line_re = re.compile(
            r"^\s*(?P<text>.+?)\s*(?P<coords>\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*\(\s*\d+\s*,\s*\d+\s*\))\s*$"
        )
        self._full_width_translation = str.maketrans({
            "（": "(",
            "）": ")",
            "，": ",",
        })

    async def predict(self, frame_idx, base64_img, task="spotting") -> tuple[int, List[SpottingItem]]:
        messages = [
            {
                "role": "system",
                "content": "",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    },
                    {
                        "type": "text",
                        "text": self._tasks[task]
                    }
                ]
            }
        ]

        spotting_list = []
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                top_p=0.95,
                seed=1234,
                stream=False,
                max_tokens=16384,
                extra_body={
                    "top_k": 1,
                    "repetition_penalty": 1.0,
                },
            )
            content = self.extract_response_text(response.choices[0].message.content)
            content = self.clean_repeated_substrings(content)
            if content != "":
                spotting_list = self.parse_spotting_response(content)
            spotting_list = self.dedup_spotting_items(spotting_list)
        except (ValidationError, LengthFinishReasonError) as e:
            # 예외가 발생해도 그냥 빈 문자열로 치환하고 로그만 남긴다
            print(f"[Warn] 프레임 {frame_idx} OCR 중 예외 발생: {e!r}")
            spotting_list = []
        except Exception as e:
            error = OcrProcessingError(frame_idx, e)
            print(f"[Error] {error}")
            raise error
        
        return frame_idx, spotting_list

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

    def normalize_spotting_response(self, response: str) -> str:
        normalized = response.translate(self._full_width_translation)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("```text", "").replace("```json", "").replace("```", "")
        return normalized.strip()

    def parse_spotting_response(self, response: str) -> List[SpottingItem]:
        normalized = self.normalize_spotting_response(response)
        spotting_list: list[SpottingItem] = []
        coord_matches = list(self._coord_pair_re.finditer(normalized))

        for line in normalized.split("\n"):
            item = self.parse_spotting_bbox_line(line)
            if item is not None:
                spotting_list.append(item)

        if spotting_list and len(spotting_list) == len(coord_matches):
            return spotting_list

        cursor = 0
        spotting_list = []
        for match in coord_matches:
            text = normalized[cursor:match.start()].split("\n")[-1].strip()
            item = self.build_spotting_item(text, match.groups())
            if item is not None:
                spotting_list.append(item)
            cursor = match.end()

        if spotting_list:
            return spotting_list

        for line in normalized.split("\n"):
            item = self.parse_spotting_line(line)
            if item is not None:
                spotting_list.append(item)

        return spotting_list

    def parse_spotting_bbox_line(self, line: str) -> Optional[SpottingItem]:
        match = self._spotting_line_re.fullmatch(line)
        if match is None:
            return None

        coord_match = self._coord_pair_re.fullmatch(match.group("coords"))
        if coord_match is None:
            return None

        return self.build_spotting_item(match.group("text"), coord_match.groups())

    def build_spotting_item(self, text: str, coords: tuple[str, str, str, str]) -> Optional[SpottingItem]:
        cleaned_text = re.sub(r"</?(?:ref|box|quad)>", "", text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip().strip("`")
        if not cleaned_text:
            return None

        x1, y1, x2, y2 = (self.clip_normalized_coord(value) for value in coords)
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        quad = (
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        )
        return SpottingItem(text=cleaned_text, quad=quad)

    def clip_normalized_coord(self, value: str | int) -> int:
        coord = int(value)
        return max(0, min(coord, 1000))

    def parse_spotting_line(self, line: str) -> Optional[SpottingItem]:
        """
        한 줄(텍스트 + 8개 LOC 토큰)을 SpottingItem(dataclass)로 파싱.

        예) "ABC<|LOC_1|>...<|LOC_8|>"
        - strict=True  : LOC가 정확히 8개가 아니면 예외
        - strict=False : 8개 이상이면 앞 8개만 사용(8개 미만이면 예외)
        """
        locs = list(map(int, self._loc_re.findall(line)))
        if len(locs) != 8:
            return None

        text = self._loc_re.split(line, maxsplit=1)[0].rstrip("\n\r")

        quad = (
            (locs[0], locs[1]),
            (locs[2], locs[3]),
            (locs[4], locs[5]),
            (locs[6], locs[7]),
        )
        return SpottingItem(text=text, quad=quad)
    
    def dedup_spotting_items(self, items: List[SpottingItem]) -> List[SpottingItem]:
        """
        완전 동일(text, quad) 중복 제거. (순서 유지)
        """
        seen: set[Tuple[str, Tuple[Tuple[int, int], ...]]] = set()
        out: List[SpottingItem] = []

        for it in items:
            key = (it.text, it.quad)  # quad까지 포함하니 '완전 동일'만 제거됨
            if key in seen:
                continue
            seen.add(key)
            out.append(it)

        return out
