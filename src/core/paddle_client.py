import re
import asyncio

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from pydantic import BaseModel, ValidationError

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
            "spotting": "Spotting:",
            "seal": "Seal Recognition:",
        }
        self._loc_re = re.compile(r"<\|LOC_(\d+)\|>")

    async def predict(self, frame_idx, base64_img, task="spotting") -> List[SpottingItem]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_img}"
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
            )
            content = response.choices[0].message.content
            if content != "":
                for line in content.split("\n"):
                    item = self.parse_spotting_line(line)
                    if isinstance(item, SpottingItem):
                        spotting_list.append(item)
        except (ValidationError, LengthFinishReasonError) as e:
            # 예외가 발생해도 그냥 빈 문자열로 치환하고 로그만 남긴다
            print(f"[Warn] 프레임 {frame_idx} OCR 중 예외 발생: {e!r}")
            spotting_list = []
        except Exception as e:
            error = OcrProcessingError(frame_idx, e)
            print(f"[Error] {error}")
            raise error
        
        return frame_idx, spotting_list

    def parse_spotting_line(self, line: str) -> SpottingItem:
        """
        한 줄(텍스트 + 8개 LOC 토큰)을 SpottingItem(dataclass)로 파싱.

        예) "ABC<|LOC_1|>...<|LOC_8|>"
        - strict=True  : LOC가 정확히 8개가 아니면 예외
        - strict=False : 8개 이상이면 앞 8개만 사용(8개 미만이면 예외)
        """
        locs = list(map(int, self._loc_re.findall(line)))
        if len(locs) != 8:
            return False

        text = self._loc_re.split(line, maxsplit=1)[0].rstrip("\n\r")

        quad = (
            (locs[0], locs[1]),
            (locs[2], locs[3]),
            (locs[4], locs[5]),
            (locs[6], locs[7]),
        )
        return SpottingItem(text=text, quad=quad)
