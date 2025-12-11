from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, List, Optional


_WS_RE = re.compile(r"\s+")


@dataclass
class Row:
    frame_number: int
    time: float
    text: str


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str


# 공백·개행·탭 등을 하나의 공백으로 줄여 주는 헬퍼
def normalize_text(text: str) -> str:
    """Strip and collapse whitespace."""
    if not text:
        return ""
    text = text.strip()
    return _WS_RE.sub(" ", text)


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


def token_set(text: str) -> set[str]:
    normalized = _strip_punct_lower_nfkc(text)
    return set(normalized.split()) if normalized else set()


def char_similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    norm_a = _strip_punct_lower_nfkc(a)
    norm_b = _strip_punct_lower_nfkc(b)
    if not norm_a and not norm_b:
        return 1.0
    return SequenceMatcher(None, norm_a, norm_b, autojunk=False).ratio()


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


def parse_csv(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, record in enumerate(reader):
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


def estimate_frame_step(times: List[float]) -> float:
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not deltas:
        return 0.04
    dt = median(deltas)
    return max(0.01, min(dt, 0.2))


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





def merge_same_text_segments(
    segments: List[Segment],
    *,
    gap: float,
    char_thresh: float = 0.94,
    token_thresh: float = 0.75,
) -> List[Segment]:
    if not segments:
        return []

    merged: List[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap_len = max(0.0, seg.start - prev.end)

        norm_prev = _strip_punct_lower_nfkc(prev.text).replace(" ", "")
        norm_curr = _strip_punct_lower_nfkc(seg.text).replace(" ", "")
        short_subset = (
            bool(norm_prev)
            and bool(norm_curr)
            and len(norm_prev) >= 5
            and len(norm_curr) <= 3
            and set(norm_curr) <= set(norm_prev)
        )
        cs = char_similarity(prev.text, seg.text)
        small_len = len(norm_prev) <= 12 and len(norm_curr) <= 12
        unique_chars = set(norm_prev + norm_curr)

        if seg.text == prev.text:
            prev_len = prev.end - prev.start
            seg_len = seg.end - seg.start
            max_len = max(prev_len, seg_len)
            if prev_len <= 2.0 and seg_len <= 2.0:
                allowed_gap = max(gap, 1.3)
            elif max_len >= 4.0 or len(norm_prev) >= 12:
                allowed_gap = max(gap, 1.6)
            else:
                allowed_gap = max(gap, 1.0)
        elif short_subset:
            allowed_gap = max(gap, 1.4)
        elif small_len and cs >= max(char_thresh - 0.06, 0.88):
            allowed_gap = max(gap, 1.0)
        elif len(unique_chars) <= 7 and cs >= max(char_thresh - 0.24, 0.70):
            allowed_gap = max(gap, 1.0)
        else:
            allowed_gap = gap

        if gap_len > allowed_gap:
            merged.append(seg)
            continue

        if seg.text == prev.text or short_subset:
            should_merge = True
        elif small_len and cs >= max(char_thresh - 0.06, 0.88):
            should_merge = True
        elif len(unique_chars) <= 7 and cs >= max(char_thresh - 0.24, 0.70):
            should_merge = True
        else:
            if cs >= char_thresh:
                should_merge = True
            elif cs >= max(char_thresh - 0.02, 0.85) and token_jaccard(prev.text, seg.text) >= token_thresh:
                should_merge = True
            else:
                should_merge = False

        if should_merge:
            if len(seg.text) > len(prev.text):
                prev.text = seg.text
            prev.end = max(prev.end, seg.end)
        else:
            merged.append(seg)

    def _squash_moan_runs(items: List[Segment]) -> List[Segment]:
        if not items:
            return []
        squashed: List[Segment] = [items[0]]
        for seg in items[1:]:
            prev = squashed[-1]
            norm_prev_local = _strip_punct_lower_nfkc(prev.text).replace(" ", "")
            norm_curr_local = _strip_punct_lower_nfkc(seg.text).replace(" ", "")
            if norm_prev_local and norm_curr_local:
                unique_local = set(norm_prev_local + norm_curr_local)
                gap_local = max(0.0, seg.start - prev.end)
                if len(unique_local) <= 7 and gap_local <= max(gap, 1.8):
                    if len(norm_curr_local) > len(norm_prev_local):
                        prev.text = seg.text
                    prev.end = max(prev.end, seg.end)
                    continue
            squashed.append(seg)
        return squashed

    return _squash_moan_runs(merged)


def build_segments(
    rows: List[Row],
    *,
    max_gap: float,
    min_duration: float,
    overlap_guard: float,
    char_thresh: float = 0.88,
    token_thresh: float = 0.60,
    similar_gap: Optional[float] = None,
    same_text_gap: float = 0.6,
    same_text_char_thresh: float = 0.94,
    same_text_token_thresh: float = 0.75,
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

    def flush(next_time: Optional[float] = None) -> None:
        nonlocal current_text, current_start, current_last_time, current_counts, current_last_seen
        if current_text is None:
            return
        end_time = current_last_time + dt
        if next_time is not None:
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

    segments = merge_same_text_segments(
        segments,
        gap=same_text_gap,
        char_thresh=same_text_char_thresh,
        token_thresh=same_text_token_thresh,
    )

    for idx, seg in enumerate(segments, start=1):
        seg.index = idx
    return segments


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
    same_text_gap: float = 0.6,
    same_text_char_thresh: float = 0.94,
    same_text_token_thresh: float = 0.75,
) -> Path:
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
        same_text_char_thresh=same_text_char_thresh,
        same_text_token_thresh=same_text_token_thresh,
    )
    if out_srt is None:
        out_srt = in_csv.with_suffix(".srt")
    write_srt(segments, out_srt)
    return out_srt


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert OCR CSV (frame_number,time,text) to SRT by merging contiguous "
            "identical or similar texts."
        )
    )
    parser.add_argument("csv", type=Path, help="Input CSV path")
    parser.add_argument("-o", "--output", type=Path, help="Output SRT path")
    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.20,
        help="Max silence (s) to keep identical text in the same cue (default: 0.20)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.30,
        help="Minimum cue duration in seconds (default: 0.30)",
    )
    parser.add_argument(
        "--overlap-guard",
        type=float,
        default=0.01,
        help="Safety margin subtracted from the next cue start to avoid overlaps",
    )
    parser.add_argument(
        "--char-thresh",
        type=float,
        default=0.88,
        help="Char-level similarity threshold for in-flight grouping (default: 0.88)",
    )
    parser.add_argument(
        "--token-thresh",
        type=float,
        default=0.60,
        help="Token Jaccard threshold for in-flight grouping (default: 0.60)",
    )
    parser.add_argument(
        "--similar-gap",
        type=float,
        default=None,
        help="Max gap to bridge for similar (non-identical) text (default: --max-gap)",
    )
    parser.add_argument(
        "--same-text-gap",
        type=float,
        default=0.6,
        help="Post-pass gap (s) to merge adjacent cues with the same text (default: 0.6)",
    )
    parser.add_argument(
        "--same-text-char-thresh",
        type=float,
        default=0.94,
        help="Char similarity needed to merge adjacent cues in the post-pass",
    )
    parser.add_argument(
        "--same-text-token-thresh",
        type=float,
        default=0.75,
        help="Token similarity needed to merge adjacent cues in the post-pass",
    )

    args = parser.parse_args(argv)

    out = convert_csv_to_srt(
        args.csv,
        args.output,
        max_gap=args.max_gap,
        min_duration=args.min_duration,
        overlap_guard=args.overlap_guard,
        char_thresh=args.char_thresh,
        token_thresh=args.token_thresh,
        similar_gap=args.similar_gap,
        same_text_gap=args.same_text_gap,
        same_text_char_thresh=args.same_text_char_thresh,
        same_text_token_thresh=args.same_text_token_thresh,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
