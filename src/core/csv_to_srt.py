from __future__ import annotations

import sys
import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, List, Optional


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


# 공백, 개행, 탭 등을 하나의 공백으로 줄여 주는 헬퍼
def normalize_text(text: str) -> str:
    """Strip and collapse whitespace."""
    if not text:
        return ""
    text = text.strip()
    return _WS_RE.sub(" ", text)


# 비교 전에 유니코드, 대소문자, 기호 차이를 줄여 문자열을 정규화
def _strip_punct_lower_nfkc(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    kept: list[str] = []
    for ch in normalized:
        if ch.isspace():
            kept.append(" ")
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            kept.append(ch)
    collapsed = _WS_RE.sub(" ", "".join(kept)).strip()
    return collapsed


# 연속으로 반복되는 문자를 줄여 비교용 문자 시그니처를 생성
def _letter_signature(normalized: str) -> str:
    if not normalized:
        return ""
    signature: list[str] = []
    prev = ""
    for ch in normalized:
        if ch.isspace():
            prev = ""
            continue
        if ch != prev:
            signature.append(ch)
        prev = ch
    return "".join(signature)


# 정규화된 문자열을 단어 집합으로 변환
def token_set(text: str) -> set[str]:
    normalized = _strip_punct_lower_nfkc(text)
    return set(normalized.split()) if normalized else set()


# 정규화된 두 문자열의 문자 단위 유사도를 계산
def char_similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    norm_a = _strip_punct_lower_nfkc(a)
    norm_b = _strip_punct_lower_nfkc(b)
    if not norm_a and not norm_b:
        return 1.0
    return SequenceMatcher(None, norm_a, norm_b, autojunk=False).ratio()


# 두 문자열의 단어 집합 겹침 정도를 자카드 계수로 계산
def token_jaccard(a: str, b: str) -> float:
    aset = token_set(a)
    bset = token_set(b)
    if not aset and not bset:
        return 1.0
    if not aset or not bset:
        return 0.0
    inter = len(aset & bset)
    union = len(aset | bset)
    return inter / union if union else 0.0


# 문자, 단어, 반복 패턴 기준을 조합해 두 자막이 같은지 판단
def are_similar(
    a: str,
    b: str,
    *,
    char_thresh: float = 0.88,
    token_thresh: float = 0.60,
) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True

    norm_a = _strip_punct_lower_nfkc(a)
    norm_b = _strip_punct_lower_nfkc(b)

    if not norm_a or not norm_b:
        return False

    letters_a = norm_a.replace(" ", "")
    letters_b = norm_b.replace(" ", "")
    avg_len = (len(letters_a) + len(letters_b)) / 2 if (letters_a or letters_b) else 0.0

    # 짧은 자막은 글자 하나 차이에도 민감하므로 길이에 따라 임계값을 조정
    def adjusted_threshold(base: float, avg: float) -> float:
        if avg <= 3:
            return min(base, 0.72)
        if avg <= 6:
            return min(base, 0.78)
        if avg <= 10:
            return min(base, 0.84)
        if avg <= 18:
            return min(base, base - 0.03)
        return base

    char_target = adjusted_threshold(char_thresh, avg_len)
    cs = char_similarity(a, b)
    if cs >= char_target:
        return True

    tj = token_jaccard(a, b)
    if tj and cs >= max(char_target - 0.08, 0.7) and tj >= token_thresh:
        return True

    # 반복 모음, 의성어처럼 문자 구성만 비슷한 경우도 한 번 더 잡아줌
    sig_a = _letter_signature(norm_a)
    sig_b = _letter_signature(norm_b)
    if sig_a and sig_b:
        shorter, longer = sorted((sig_a, sig_b), key=len)
        len_longer = len(longer)
        if len_longer <= 10:
            if shorter == longer:
                return True
            if shorter in longer and len_longer - len(shorter) <= 3:
                return True

        from difflib import SequenceMatcher

        sig_ratio = SequenceMatcher(None, sig_a, sig_b, autojunk=False).ratio()
        if sig_ratio >= max(char_target - 0.1, 0.78):
            return True

        letters_overlap = set(sig_a) & set(sig_b)
        union = set(sig_a) | set(sig_b)
        if union:
            overlap_ratio = len(letters_overlap) / len(union)
            if overlap_ratio >= 0.75 and abs(len(sig_a) - len(sig_b)) <= 3 and len_longer <= 12:
                return True

    return False


# OCR CSV를 읽어 시간순 Row 목록으로 정리
def parse_csv(path: Path) -> List[Row]:
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
                    or 0.0
                )
            except Exception:
                continue
            text = record.get("text") or ""
            rows.append(Row(frame_number=frame, time=timestamp, text=text))
    rows.sort(key=lambda r: r.time)
    return rows


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


# 같은 자막으로 판단되는 인접 구간을 다시 병합
def merge_same_text_segments(
    segments: List[Segment],
    *,
    gap: float,
) -> List[Segment]:
    if not segments:
        return []
    # 정규화 결과가 같은 자막은 최대 1초 간격까지 다시 합침
    normalized_merge_gap = max(gap, 1.0)
    merged: List[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap_len = max(0.0, seg.start - prev.end)
        norm_prev = _strip_punct_lower_nfkc(prev.text)
        norm_curr = _strip_punct_lower_nfkc(seg.text)
        if norm_prev and norm_prev == norm_curr and gap_len <= normalized_merge_gap:
            if len(seg.text) > len(prev.text):
                prev.text = seg.text
            prev.end = max(prev.end, seg.end)
        else:
            merged.append(seg)
    return merged


# OCR 행 목록을 시간 흐름에 따라 자막 구간으로 묶어 생성
def build_segments(
    rows: List[Row],
    *,
    max_gap: float,
    min_duration: float,
    overlap_guard: float,
    char_thresh: float = 0.88,
    token_thresh: float = 0.60,
    similar_gap: Optional[float] = None,
    same_text_gap: float = 1.0,
) -> List[Segment]:
    if not rows:
        return []

    times = [row.time for row in rows]
    dt = estimate_frame_step(times)
    segments: List[Segment] = []

    current_text: Optional[str] = None
    current_start = 0.0
    current_last_time = 0.0
    current_counts: dict[str, int] = {}
    current_last_seen: dict[str, float] = {}

    # 현재 누적 중인 자막 구간을 대표 텍스트 하나로 확정해 저장
    def flush(next_time: Optional[float] = None) -> None:
        nonlocal current_text, current_start, current_last_time, current_counts, current_last_seen
        if current_text is None:
            return
        end_time = current_last_time + dt
        if next_time is not None:
            # 다음 자막 시작 직전까지만 잘라 겹침을 방지
            end_time = min(end_time, max(current_start, next_time - overlap_guard))
        if current_counts:
            max_count = max(current_counts.values())
            candidates = [t for t, c in current_counts.items() if c == max_count]
            if len(candidates) > 1:
                rep = max(
                    candidates,
                    key=lambda t: (current_last_seen.get(t, -1.0), len(t))
                )
            else:
                rep = candidates[0]
        else:
            rep = current_text
        rep = normalize_text(rep)
        if rep and end_time - current_start >= min_duration:
            segments.append(
                Segment(
                    index=len(segments) + 1,
                    start=current_start,
                    end=end_time,
                    text=rep,
                )
            )
        current_text = None
        current_counts = {}
        current_last_seen = {}

    for row in rows:
        tnorm = normalize_text(row.text)
        if not tnorm:
            continue
        if current_text is None:
            current_text = tnorm
            current_start = row.time
            current_last_time = row.time
            current_counts[tnorm] = current_counts.get(tnorm, 0) + 1
            current_last_seen[tnorm] = row.time
            continue

        gap = row.time - current_last_time
        allow_same = max_gap
        allow_similar = similar_gap if similar_gap is not None else max_gap

        if tnorm == current_text or are_similar(
            current_text,
            tnorm,
            char_thresh=char_thresh,
            token_thresh=token_thresh,
        ):
            if (tnorm == current_text and gap <= allow_same) or (
                tnorm != current_text and gap <= allow_similar
            ):
                current_last_time = row.time
                current_counts[tnorm] = current_counts.get(tnorm, 0) + 1
                current_last_seen[tnorm] = row.time
            else:
                flush(next_time=row.time)
                current_text = tnorm
                current_start = row.time
                current_last_time = row.time
                current_counts = {tnorm: 1}
                current_last_seen = {tnorm: row.time}
        else:
            flush(next_time=row.time)
            current_text = tnorm
            current_start = row.time
            current_last_time = row.time
            current_counts = {tnorm: 1}
            current_last_seen = {tnorm: row.time}

    flush(next_time=None)

    # 1차 구간화 후에도 정규화 결과가 같은 자막은 1초 이내에서 다시 합침
    segments = merge_same_text_segments(
        segments,
        gap=same_text_gap,
    )

    for idx, seg in enumerate(segments, start=1):
        seg.index = idx
    return segments


# 완성된 자막 구간 목록을 SRT 파일로 기록
def write_srt(segments: Iterable[Segment], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        first = True
        for seg in segments:
            if not first:
                handle.write("\n")
            first = False
            handle.write(f"{seg.index}\n")
            handle.write(
                f"{to_srt_timestamp(seg.start)} --> {to_srt_timestamp(seg.end)}\n"
            )
            handle.write(f"{seg.text}\n")


# CSV 입력을 읽어 SRT 파일로 변환하는 진입 함수
def convert_csv_to_srt(
    in_csv: Path,
    out_srt: Optional[Path] = None,
    *,
    max_gap: float = 0.20,
    min_duration: float = 0.30,
    overlap_guard: float = 0.01,
    char_thresh: float = 0.88,
    token_thresh: float = 0.60,
    similar_gap: Optional[float] = None,
    same_text_gap: float = 1.0,
) -> Path:
    # CSV 파싱 -> 구간 병합 -> SRT 저장 순서로 변환 수행
    rows = parse_csv(in_csv)
    segments = build_segments(
        rows,
        max_gap=max_gap,
        min_duration=min_duration,
        overlap_guard=overlap_guard,
        char_thresh=char_thresh,
        token_thresh=token_thresh,
        similar_gap=similar_gap,
        same_text_gap=same_text_gap,
    )
    if out_srt is None:
        out_srt = in_csv.with_suffix(".srt")
    write_srt(segments, out_srt)
    return out_srt


# 명령행 인자를 받아 CSV -> SRT 변환을 실행
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert OCR CSV (frame_number,time,text) to SRT by merging contiguous "
            "identical or similar texts."
        )
    )
    parser.add_argument("csv", type=Path, help="Input CSV path")
    parser.add_argument("-o", "--output", type=Path, help="Output SRT path")

    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    out = convert_csv_to_srt(
        args.csv,
        args.output
    )
    print(out)


if __name__ == "__main__":
    main()
