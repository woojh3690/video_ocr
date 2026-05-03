from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple


class OcrProcessingError(RuntimeError):
    def __init__(self, frame_number: int, original_exception: Exception):
        self.frame_number = frame_number
        self.original_exception = original_exception
        message = f"프레임 {frame_number} OCR 처리 중 복구할 수 없는 오류: {original_exception}"
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
            raise ValueError(f"quad 형식이 올바르지 않습니다: {quad_raw!r}")

        quad = tuple((int(point[0]), int(point[1])) for point in quad_raw)  # type: ignore[arg-type]
        if len(quad) != 4:
            raise ValueError("quad 좌표는 4개여야 합니다.")

        return cls(text=str(data["text"]), quad=quad)  # type: ignore[arg-type]
