from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any, Iterable, List, Optional


_WS_RE = re.compile(r"\s+")


# OCR CSV 한 줄을 내부 처리용 구조체로 보관
@dataclass
class Row:
    frame_number: int
    time: float
    text: str


# SRT 한 구간을 나타내는 구조체
@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str


# 분할 전에 필요한 문자열과 상태를 함께 보관
@dataclass
class PreparedRow:
    frame_number: int
    time: float
    display_text: str
    compare_text: str
    is_blank: bool
    is_noise: bool = False


# 현재 누적 중인 자막 후보를 보관
@dataclass
class ClusterState:
    start_index: int
    start_time: float
    last_index: int
    last_time: float
    row_indices: List[int] = field(default_factory=list)
    display_counts: Counter[str] = field(default_factory=Counter)
    compare_counts: Counter[str] = field(default_factory=Counter)
    display_last_seen: dict[str, float] = field(default_factory=dict)
    compare_last_seen: dict[str, float] = field(default_factory=dict)
    last_display_text: str = ""
    last_compare_text: str = ""


# 공백, 개행, 탭 등을 하나의 공백으로 줄여 주는 헬퍼
def normalize_text(text: str) -> str:
    """공백만 정리한 출력용 문자열을 만든다."""
    if not text:
        return ""
    return _WS_RE.sub(" ", text.strip())


# 비교 전에 유니코드와 기호 차이를 줄여 비교용 문자열을 정규화
def normalize_compare_text(text: str) -> str:
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    kept: list[str] = []
    for ch in normalized:
        if ch.isalnum():
            kept.append(ch)
        elif ch.isspace():
            kept.append(" ")
    return _WS_RE.sub(" ", "".join(kept)).strip()


# 정규화된 두 문자열의 문자 단위 유사도를 계산
def score_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


# 정규화된 두 문자열의 문자 차이 개수와 비율을 계산
def measure_text_difference(left: str, right: str) -> tuple[int, float]:
    compact_left = left.replace(" ", "")
    compact_right = right.replace(" ", "")
    if not compact_left and not compact_right:
        return 0, 0.0

    diff_count = abs(len(compact_left) - len(compact_right))
    diff_count += sum(
        1
        for left_ch, right_ch in zip(compact_left, compact_right)
        if left_ch != right_ch
    )
    base_length = max(len(compact_left), len(compact_right), 1)
    return diff_count, diff_count / base_length


# 길이에 따라 유사도 기준을 달리해 같은 자막인지 판단
def texts_are_similar(left: str, right: str) -> bool:
    compact_left = left.replace(" ", "")
    compact_right = right.replace(" ", "")
    if not compact_left or not compact_right:
        return False
    if compact_left == compact_right:
        return True

    similarity = score_similarity(compact_left, compact_right)
    diff_count, diff_ratio = measure_text_difference(compact_left, compact_right)
    max_len = max(len(compact_left), len(compact_right))

    if max_len <= 6:
        return diff_count <= 2 or similarity >= 0.72
    if max_len <= 12:
        return diff_count <= 3 or diff_ratio <= 0.30 or similarity >= 0.80
    return diff_ratio <= 0.24 or similarity >= 0.88


# 딕셔너리 안에서 원하는 키를 재귀적으로 찾는다
def find_nested_value(record: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(record, dict):
        for key in keys:
            value = record.get(key)
            if value not in (None, "", [], {}):
                return value
        for value in record.values():
            found = find_nested_value(value, keys)
            if found not in (None, "", [], {}):
                return found
    if isinstance(record, list):
        for item in record:
            found = find_nested_value(item, keys)
            if found not in (None, "", [], {}):
                return found
    return None


# 다양한 JSON 형태에서 텍스트를 최대한 문자열로 변환
def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [coerce_text(item) for item in value]
        return " ".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "texts", "content", "caption", "value"):
            text = coerce_text(value.get(key))
            if text:
                return text
    return str(value)


# CSV를 읽어 시간순 Row 목록으로 정리
def parse_csv_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, record in enumerate(reader):
            # CSV 컬럼명이 조금 달라도 최대한 흡수해서 읽음
            try:
                frame = int(
                    record.get("frame_number")
                    or record.get("frame")
                    or record.get("index")
                    or idx
                )
            except Exception:
                frame = idx

            try:
                timestamp = float(
                    record.get("time")
                    or record.get("seconds")
                    or record.get("t")
                    or (idx * 0.033)
                )
            except Exception:
                timestamp = idx * 0.033

            text = record.get("text") or record.get("texts") or ""
            rows.append(Row(frame_number=frame, time=timestamp, text=text))

    rows.sort(key=lambda row: row.time)
    return rows


# JSONL도 같은 Row 목록으로 읽어 범용 입력으로 처리
def parse_jsonl_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception:
                continue

            frame_value = find_nested_value(record, ("frame_number", "frame", "index", "id"))
            time_value = find_nested_value(record, ("time", "seconds", "t", "timestamp"))
            text_value = find_nested_value(record, ("text", "texts", "caption", "content"))

            try:
                frame = int(frame_value) if frame_value is not None else idx
            except Exception:
                frame = idx

            try:
                timestamp = float(time_value) if time_value is not None else idx * 0.033
            except Exception:
                timestamp = idx * 0.033

            text = coerce_text(text_value)
            rows.append(Row(frame_number=frame, time=timestamp, text=text))

    rows.sort(key=lambda row: row.time)
    return rows


# 입력 포맷을 자동 감지해 공통 Row 목록으로 읽는다
def parse_csv(path: Path) -> List[Row]:
    if path.suffix.lower() == ".jsonl":
        return parse_jsonl_rows(path)
    return parse_csv_rows(path)


# 프레임 간격의 대표값을 계산해 자막 종료 보정에 사용
def estimate_frame_step(times: List[float]) -> float:
    # 프레임 간격의 중앙값을 사용해 자막 종료 시점을 안정적으로 추정
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not deltas:
        return 0.04
    dt = median(deltas)
    return max(0.01, min(dt, 0.2))


# 초 단위 시간을 SRT 타임스탬프 문자열로 변환
def to_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# OCR 행을 출력용 문자열과 비교용 문자열로 미리 정리
def prepare_rows(rows: List[Row]) -> List[PreparedRow]:
    prepared_rows: List[PreparedRow] = []
    for row in rows:
        display_text = normalize_text(row.text)
        compare_text = normalize_compare_text(row.text)
        prepared_rows.append(
            PreparedRow(
                frame_number=row.frame_number,
                time=row.time,
                display_text=display_text,
                compare_text=compare_text,
                is_blank=display_text == "",
            )
        )
    return prepared_rows


# 짧고 자주 반복되는 ASCII 표시는 워터마크로 보고 제외
def is_probable_noise_text(text: str, max_len: int) -> bool:
    compact = text.replace(" ", "")
    if not compact or len(compact) > max_len:
        return False
    return compact.isascii() and compact.upper() == compact


# 자주 반복되는 워터마크 후보를 노이즈로 표시
def mark_noise_rows(
    prepared_rows: List[PreparedRow],
    *,
    noise_text_max_len: int,
    noise_frequency_ratio: float,
) -> None:
    candidates = [row.display_text for row in prepared_rows if row.display_text]
    if not candidates:
        return

    counts = Counter(candidates)
    total = len(candidates)
    noise_texts = {
        text
        for text, count in counts.items()
        if is_probable_noise_text(text, noise_text_max_len)
        and (count / total) >= noise_frequency_ratio
    }

    for row in prepared_rows:
        row.is_noise = row.display_text in noise_texts


# 카운터에서 대표 문자열 하나를 안정적으로 고른다
def pick_best_text(counts: Counter[str], last_seen: dict[str, float]) -> str:
    return max(
        counts,
        key=lambda text: (counts[text], last_seen.get(text, -1.0), len(text)),
    )


# 새 자막 후보를 시작한다
def start_cluster(row_index: int, row: PreparedRow) -> ClusterState:
    cluster = ClusterState(
        start_index=row_index,
        start_time=row.time,
        last_index=row_index,
        last_time=row.time,
    )
    add_row_to_cluster(cluster, row_index, row)
    return cluster


# 현재 자막 후보에 row 하나를 추가한다
def add_row_to_cluster(cluster: ClusterState, row_index: int, row: PreparedRow) -> None:
    cluster.row_indices.append(row_index)
    cluster.last_index = row_index
    cluster.last_time = row.time
    cluster.last_display_text = row.display_text
    cluster.last_compare_text = row.compare_text

    if row.display_text:
        cluster.display_counts[row.display_text] += 1
        cluster.display_last_seen[row.display_text] = row.time
    if row.compare_text:
        cluster.compare_counts[row.compare_text] += 1
        cluster.compare_last_seen[row.compare_text] = row.time


# 현재 자막 후보의 비교용 대표 문자열을 고른다
def pick_cluster_compare_text(cluster: ClusterState) -> str:
    if not cluster.compare_counts:
        return ""
    return pick_best_text(cluster.compare_counts, cluster.compare_last_seen)


# 현재 자막 후보의 출력용 대표 문자열을 고른다
def pick_cluster_display_text(cluster: ClusterState) -> str:
    if not cluster.display_counts:
        return ""
    return pick_best_text(cluster.display_counts, cluster.display_last_seen)


# 새 후보가 충분히 안정적으로 반복됐는지 판단한다
def cluster_is_confirmed(
    cluster: ClusterState,
    *,
    confirm_rows: int,
    confirm_duration: float,
) -> bool:
    return (
        len(cluster.row_indices) >= confirm_rows
        or (cluster.last_time - cluster.start_time) >= confirm_duration
    )


# row가 현재 자막 후보에 붙을 수 있는지 판단한다
def row_matches_cluster(cluster: ClusterState, row: PreparedRow) -> bool:
    anchor_text = pick_cluster_compare_text(cluster)
    if anchor_text and texts_are_similar(anchor_text, row.compare_text):
        return True
    if cluster.last_compare_text and texts_are_similar(cluster.last_compare_text, row.compare_text):
        return True
    return False


# 현재 자막 후보를 실제 세그먼트로 확정해 저장한다
def flush_cluster(
    cluster: Optional[ClusterState],
    *,
    prepared_rows: List[PreparedRow],
    segments: List[Segment],
    frame_step: float,
    min_duration: float,
    end_cap: Optional[float],
) -> None:
    if cluster is None or not cluster.row_indices:
        return

    text = pick_cluster_display_text(cluster)
    if not text:
        return

    start_time = cluster.start_time
    end_time = cluster.last_time + frame_step
    if end_cap is not None:
        end_time = min(end_time, max(start_time, end_cap))

    if end_time - start_time < min_duration:
        return

    segments.append(
        Segment(
            index=len(segments) + 1,
            start=start_time,
            end=end_time,
            text=text,
        )
    )


# 같은 자막이 짧게 끊긴 경우 마지막으로 한 번 더 병합한다
def merge_adjacent_segments(segments: List[Segment], merge_gap: float) -> List[Segment]:
    if not segments:
        return []

    merged: List[Segment] = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        gap = max(0.0, segment.start - previous.end)
        prev_compare = normalize_compare_text(previous.text)
        curr_compare = normalize_compare_text(segment.text)
        if prev_compare and prev_compare == curr_compare and gap <= merge_gap:
            if len(segment.text) > len(previous.text):
                previous.text = segment.text
            previous.end = max(previous.end, segment.end)
            continue
        merged.append(segment)

    for index, segment in enumerate(merged, start=1):
        segment.index = index
    return merged


# OCR 행 목록을 시간 흐름에 따라 자막 구간으로 묶어 생성
def build_segments(
    rows: List[Row],
    *,
    min_duration: float,
    overlap_guard: float,
    blank_streak_split: int = 3,
    confirm_rows: int = 3,
    confirm_duration: float = 0.12,
    merge_gap: float = 0.45,
    noise_text_max_len: int = 4,
    noise_frequency_ratio: float = 0.02,
) -> List[Segment]:
    if not rows:
        return []

    frame_step = estimate_frame_step([row.time for row in rows])
    prepared_rows = prepare_rows(rows)
    mark_noise_rows(
        prepared_rows,
        noise_text_max_len=noise_text_max_len,
        noise_frequency_ratio=noise_frequency_ratio,
    )

    segments: List[Segment] = []
    current_cluster: Optional[ClusterState] = None
    pending_cluster: Optional[ClusterState] = None
    blank_count = 0
    blank_start_time: Optional[float] = None

    for row_index, row in enumerate(prepared_rows):
        if row.is_blank:
            blank_count += 1
            if blank_start_time is None:
                blank_start_time = row.time

            if blank_count >= blank_streak_split:
                if pending_cluster and cluster_is_confirmed(
                    pending_cluster,
                    confirm_rows=confirm_rows,
                    confirm_duration=confirm_duration,
                ):
                    flush_cluster(
                        current_cluster,
                        prepared_rows=prepared_rows,
                        segments=segments,
                        frame_step=frame_step,
                        min_duration=min_duration,
                        end_cap=pending_cluster.start_time - overlap_guard,
                    )
                    current_cluster = pending_cluster

                flush_cluster(
                    current_cluster,
                    prepared_rows=prepared_rows,
                    segments=segments,
                    frame_step=frame_step,
                    min_duration=min_duration,
                    end_cap=blank_start_time - overlap_guard,
                )
                current_cluster = None
                pending_cluster = None
            continue

        blank_count = 0
        blank_start_time = None

        if row.is_noise or not row.compare_text:
            continue

        if current_cluster is None:
            current_cluster = start_cluster(row_index, row)
            continue

        if row_matches_cluster(current_cluster, row):
            add_row_to_cluster(current_cluster, row_index, row)
            pending_cluster = None
            continue

        if pending_cluster is None:
            pending_cluster = start_cluster(row_index, row)
        elif row_matches_cluster(pending_cluster, row):
            add_row_to_cluster(pending_cluster, row_index, row)
        else:
            pending_cluster = start_cluster(row_index, row)

        if cluster_is_confirmed(
            pending_cluster,
            confirm_rows=confirm_rows,
            confirm_duration=confirm_duration,
        ):
            flush_cluster(
                current_cluster,
                prepared_rows=prepared_rows,
                segments=segments,
                frame_step=frame_step,
                min_duration=min_duration,
                end_cap=pending_cluster.start_time - overlap_guard,
            )
            current_cluster = pending_cluster
            pending_cluster = None

    if pending_cluster and cluster_is_confirmed(
        pending_cluster,
        confirm_rows=confirm_rows,
        confirm_duration=confirm_duration,
    ):
        flush_cluster(
            current_cluster,
            prepared_rows=prepared_rows,
            segments=segments,
            frame_step=frame_step,
            min_duration=min_duration,
            end_cap=pending_cluster.start_time - overlap_guard,
        )
        current_cluster = pending_cluster

    flush_cluster(
        current_cluster,
        prepared_rows=prepared_rows,
        segments=segments,
        frame_step=frame_step,
        min_duration=min_duration,
        end_cap=None,
    )
    return merge_adjacent_segments(segments, merge_gap)


# 완성된 자막 구간 목록을 SRT 파일로 기록
def write_srt(segments: Iterable[Segment], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        first = True
        for segment in segments:
            if not first:
                handle.write("\n")
            first = False
            handle.write(f"{segment.index}\n")
            handle.write(
                f"{to_srt_timestamp(segment.start)} --> {to_srt_timestamp(segment.end)}\n"
            )
            handle.write(f"{segment.text}\n")


# 입력을 읽어 SRT 파일로 변환하는 진입 함수
def convert_csv_to_srt(
    in_csv: Path,
    out_srt: Optional[Path] = None,
    *,
    min_duration: float = 0.30,
    overlap_guard: float = 0.01,
    blank_streak_split: int = 3,
    confirm_rows: int = 3,
    confirm_duration: float = 0.12,
    merge_gap: float = 0.45,
    noise_text_max_len: int = 4,
    noise_frequency_ratio: float = 0.02,
) -> Path:
    # 입력 파싱 -> 안정 구간 분할 -> SRT 저장 순서로 변환 수행
    rows = parse_csv(in_csv)
    segments = build_segments(
        rows,
        min_duration=min_duration,
        overlap_guard=overlap_guard,
        blank_streak_split=blank_streak_split,
        confirm_rows=confirm_rows,
        confirm_duration=confirm_duration,
        merge_gap=merge_gap,
        noise_text_max_len=noise_text_max_len,
        noise_frequency_ratio=noise_frequency_ratio,
    )
    if out_srt is None:
        out_srt = in_csv.with_suffix(".srt")
    write_srt(segments, out_srt)
    return out_srt


# 명령행 인자를 받아 입력 -> SRT 변환을 실행
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert OCR CSV or JSONL to SRT by tracking stable subtitle spans "
            "and confirming changes only after they persist."
        )
    )
    parser.add_argument("csv", type=Path, help="Input CSV or JSONL path")
    parser.add_argument("-o", "--output", type=Path, help="Output SRT path")

    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    out = convert_csv_to_srt(
        args.csv,
        args.output,
    )
    print(out)


if __name__ == "__main__":
    main()
