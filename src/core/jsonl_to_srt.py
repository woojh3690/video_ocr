import sys
import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from typing import List

import cv2
import numpy as np
import unicodedata
from norfair import Detection, Tracker
from norfair.tracker import TrackedObject
from PIL import Image, ImageDraw, ImageFont
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from core.paddle_client import SpottingItem

@dataclass
class FrameInfo:
    frame_idx: int
    timestamp_sec: float
    spotting_items: List[SpottingItem]
    track_ids: list[int] = field(default_factory=list)


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str


def fine_video(jsonl_path_obj: Path) -> Path:
    for file in jsonl_path_obj.parent.iterdir():
        if file.stem == jsonl_path_obj.stem and file.suffix not in [".jsonl", ".srt"]:
            return file
    raise FileNotFoundError("Video file not found")


def _track_color(track_id: int) -> tuple[int, int, int]:
    # 처리 로직 주석
    seed = (track_id * 2654435761) & 0xFFFFFFFF
    b = 64 + (seed & 0x7F)
    g = 64 + ((seed >> 8) & 0x7F)
    r = 64 + ((seed >> 16) & 0x7F)
    return int(b), int(g), int(r)


def _clip_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    # 처리 로직 주석
    nx1 = max(0, min(x1, width - 1))
    ny1 = max(0, min(y1, height - 1))
    nx2 = max(0, min(x2, width - 1))
    ny2 = max(0, min(y2, height - 1))
    if nx2 < nx1:
        nx1, nx2 = nx2, nx1
    if ny2 < ny1:
        ny1, ny2 = ny2, ny1
    return nx1, ny1, nx2, ny2


def _scaled_bbox_from_item(spotting_item: SpottingItem, width: int, height: int, norm_coord_max = 1000) -> tuple[int, int, int, int]:
    # 정규화 좌표(0~1000)를 프레임 픽셀 좌표로 변환
    x1, y1, x2, y2 = spotting_item.bbox
    if width <= 1 or height <= 1:
        return 0, 0, 0, 0

    sx1 = int(round((x1 / norm_coord_max) * (width - 1)))
    sy1 = int(round((y1 / norm_coord_max) * (height - 1)))
    sx2 = int(round((x2 / norm_coord_max) * (width - 1)))
    sy2 = int(round((y2 / norm_coord_max) * (height - 1)))
    return _clip_bbox(sx1, sy1, sx2, sy2, width, height)


def _load_font(font_size: int) -> ImageFont.ImageFont:
    # 처리 로직 주석
    for font_path in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc"):
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    if not text:
        return 0
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return max(0, right - left)


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    _, top, _, bottom = draw.textbbox((0, 0), "?A", font=font)
    return max(1, bottom - top + 2)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    # 처리 로직 주석
    if max_width <= 0:
        return []

    wrapped_lines: list[str] = []
    raw_lines = text.splitlines() if text else [""]
    for raw_line in raw_lines:
        if raw_line == "":
            wrapped_lines.append("")
            continue

        current = ""
        for ch in raw_line:
            candidate = current + ch
            if _text_width(draw, candidate, font) <= max_width or current == "":
                current = candidate
                continue
            wrapped_lines.append(current)
            current = ch
        wrapped_lines.append(current)
    return wrapped_lines


def _draw_text_inside_boxes(panel_bgr: np.ndarray, frame_info: FrameInfo) -> np.ndarray:
    # 처리 로직 주석
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(panel_rgb)
    draw = ImageDraw.Draw(image)

    for spotting_item, track_id in zip(frame_info.spotting_items, frame_info.track_ids):
        if track_id is None:
            continue

        x1, y1, x2, y2 = _scaled_bbox_from_item(spotting_item, panel_bgr.shape[1], panel_bgr.shape[0])
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 4 or box_h <= 4:
            continue

        # 처리 로직 주석
        font_size = max(12, min(24, box_h // 4))
        font = _load_font(font_size)
        line_h = _line_height(draw, font)
        max_lines = max(1, box_h // line_h)

        lines = _wrap_text(draw, spotting_item.text, font, max_width=max(1, box_w - 4))
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if lines:
                # 처리 로직 주석
                while lines[-1] and _text_width(draw, lines[-1] + "...", font) > max(1, box_w - 4):
                    lines[-1] = lines[-1][:-1]
                lines[-1] = lines[-1] + "..."

        text_color = tuple(int(c) for c in _track_color(track_id)[::-1])
        ty = y1 + 2
        for line in lines:
            draw.text((x1 + 2, ty), line, font=font, fill=text_color)
            ty += line_h
            if ty > y2:
                break

    rendered_rgb = np.array(image)
    return cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)


def _render_frame_pair(frame_bgr: np.ndarray, frame_info: FrameInfo) -> np.ndarray:
    # 처리 로직 주석
    left_panel = frame_bgr.copy()
    right_panel = np.full_like(frame_bgr, 255)

    panel_h, panel_w = frame_bgr.shape[:2]
    for spotting_item, track_id in zip(frame_info.spotting_items, frame_info.track_ids):
        if track_id is None:
            continue

        x1, y1, x2, y2 = _scaled_bbox_from_item(spotting_item, panel_w, panel_h)
        if x2 <= x1 or y2 <= y1:
            continue

        color = _track_color(track_id)
        cv2.rectangle(left_panel, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(right_panel, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_x1 = x1
        label_y1 = max(0, y1 - th - 8)
        label_x2 = min(panel_w - 1, label_x1 + tw + 8)
        label_y2 = min(panel_h - 1, label_y1 + th + 8)
        cv2.rectangle(right_panel, (label_x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            right_panel,
            label,
            (label_x1 + 4, min(panel_h - 1, label_y2 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    right_panel = _draw_text_inside_boxes(right_panel, frame_info)
    return np.hstack((left_panel, right_panel))


def _dump_visualized_frames(video_path: Path, frame_infos: list[FrameInfo], output_dir: Path) -> None:
    # 처리 로직 주석
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Viz] 출력 폴더: {output_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video file: {video_path}")

    sorted_infos = sorted(frame_infos, key=lambda info: info.frame_idx)
    if not sorted_infos:
        cap.release()
        print("[Viz] 저장할 프레임이 없습니다.")
        return

    target_idx = 0
    saved_count = 0
    skipped_count = 0

    while target_idx < len(sorted_infos):
        ret, frame = cap.read()
        if not ret:
            skipped_count += len(sorted_infos) - target_idx
            break

        current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        target_frame_idx = sorted_infos[target_idx].frame_idx

        if current_frame_idx < target_frame_idx:
            continue

        if current_frame_idx > target_frame_idx:
            while target_idx < len(sorted_infos) and sorted_infos[target_idx].frame_idx < current_frame_idx:
                print(f"[Viz][Warn] 영상에서 프레임 {sorted_infos[target_idx].frame_idx}를 찾지 못해 건너뜁니다.")
                skipped_count += 1
                target_idx += 1
            continue

        while target_idx < len(sorted_infos) and sorted_infos[target_idx].frame_idx == current_frame_idx:
            composed = _render_frame_pair(frame, sorted_infos[target_idx])
            out_path = output_dir / f"frame_{current_frame_idx:06d}.jpg"
            ok = cv2.imwrite(str(out_path), composed, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                saved_count += 1
            else:
                print(f"[Viz][Warn] 프레임 저장 실패: {out_path}")
                skipped_count += 1
            target_idx += 1

    cap.release()
    print(f"[Viz] 저장 완료: {saved_count}개, 건너뜀: {skipped_count}개")


def jsonl_to_srt(jsonl_path_obj: Path, visualize=False):
    # 경로 확인
    if not jsonl_path_obj.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path_obj}")
    if visualize:
        vd_file = fine_video(jsonl_path_obj)

    # JSONL 파일 읽기
    ocr_results: list[FrameInfo] = []
    with jsonl_path_obj.open("r", encoding="utf-8") as handle:
        while handle:
            line = handle.readline()
            if not line:
                break
            ocr_res_json = json.loads(line)
            frame_idx = int(ocr_res_json["frame_number"])
            timestamp_sec = float(ocr_res_json["time"])
            spotting_items = []
            for item in ocr_res_json["spotting_items"]:
                spotting_items.append(SpottingItem.from_dict(item))
            ocr_results.append(FrameInfo(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                spotting_items=spotting_items,
            ))

    # 글자(유니코드 Letter) 개수가 1 이하인 OCR 결과 제거
    for frame_info in ocr_results:
        if not frame_info.spotting_items:
            continue
        
        keep_indices = []
        for idx, item in enumerate(frame_info.spotting_items):
            text = item.text
            letter_count = sum(1 for ch in text if unicodedata.category(ch).startswith("L"))
            if letter_count > 1:
                keep_indices.append(idx)

        # 인덱스가 유지되는 항목만 남기도록 필터링
        if len(keep_indices) != len(frame_info.spotting_items):
            frame_info.spotting_items = [frame_info.spotting_items[idx] for idx in keep_indices]

    def normalize_text(text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text).casefold()
        return "".join(ch for ch in text if ch.isalnum())

    def box_from_points(points: np.ndarray) -> tuple[float, float, float, float]:
        x1 = float(min(points[0][0], points[1][0]))
        y1 = float(min(points[0][1], points[1][1]))
        x2 = float(max(points[0][0], points[1][0]))
        y2 = float(max(points[0][1], points[1][1]))
        return x1, y1, x2, y2

    def iou_from_points(candidate_points: np.ndarray, tracked_points: np.ndarray) -> float:
        cx1, cy1, cx2, cy2 = box_from_points(candidate_points)
        tx1, ty1, tx2, ty2 = box_from_points(tracked_points)

        inter_w = max(0.0, min(cx2, tx2) - max(cx1, tx1))
        inter_h = max(0.0, min(cy2, ty2) - max(cy1, ty1))
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0

        area_c = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
        area_t = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
        union = area_c + area_t - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    # 트래커 파라미터 초기화
    text_weight_base = 0.24
    max_text_weight_at_conf = 0.90
    force_match_similarity = 0.30
    strong_mismatch_similarity = 0.35
    mismatch_penalty = 0.08
    
    def hybrid_distance(det: Detection, tracked_object: TrackedObject) -> float:
        iou_score = iou_from_points(det.points, tracked_object.estimate)
        iou_distance = 1.0 - iou_score

        det_data = det.data or {}
        trk_data = tracked_object.last_detection.data if tracked_object.last_detection else {}
        trk_data = trk_data or {}

        det_text = det_data.get("norm_text", "")
        trk_text = trk_data.get("norm_text", "")
        conf_factor = 1.0
        if det_text and trk_text:
            similarity = SequenceMatcher(None, det_text, trk_text).ratio()
            if similarity <= force_match_similarity:
                return 1.0
            text_distance = 1.0 - similarity

            scaled = min(1.0, conf_factor / max_text_weight_at_conf)
            text_weight = min(text_weight_base, text_weight_base * max(0.0, scaled))
            distance = (1.0 - text_weight) * iou_distance + text_weight * text_distance

            if similarity <= strong_mismatch_similarity:
                distance = min(1.0, distance + mismatch_penalty)
            return distance

        return iou_distance

    tracker = Tracker(
        distance_function=hybrid_distance,
        distance_threshold=0.7,
        initialization_delay=0,
        hit_counter_max=4,
    )

    # 트래커를 사용해 텍스트 박스별 track id 계산
    for frame_info in ocr_results:
        rec_boxes = [item.bbox for item in frame_info.spotting_items]
        rec_texts = [item.text for item in frame_info.spotting_items]
        if len(rec_boxes) == 0:
            tracker.update(detections=[])
            frame_info.track_ids = []
            continue
        
        detections: list[Detection] = []
        for det_idx, box in enumerate(rec_boxes):
            x1, y1, x2, y2 = box
            text = rec_texts[det_idx] if det_idx < len(rec_texts) else ""
            detections.append(Detection(
                points=np.array([[x1, y1], [x2, y2]]),
                data={
                    "det_index": det_idx,
                    "raw_text": text,
                    "norm_text": normalize_text(text),
                }
            ))

        # 현재 프레임 track id 슬롯 준비
        frame_info.track_ids = [None] * len(rec_boxes)
        tracked_objects = tracker.update(detections=detections)

        # 이번 프레임에서 생성된 detection만 사용
        curr_det_ids = {id(d) for d in detections}

        for t in tracked_objects:
            if id(t.last_detection) not in curr_det_ids:
                continue  # 이전 프레임 last_detection 기반 트랙은 제외
            det_index = t.last_detection.data["det_index"]
            frame_info.track_ids[det_index] = t.id

        if any(track_id is None for track_id in frame_info.track_ids):
            raise RuntimeError(
                f"Unmatched detection found at frame {frame_info.frame_idx}: {frame_info.track_ids}"
            )

    # 종료 시점이 가깝고 시작 시점도 가까운 트랙은 동일 자막으로 간주
    first_frame_by_track: dict[int, int] = {}
    first_time_by_track: dict[int, float] = {}
    last_frame_by_track: dict[int, int] = {}

    # track_id 기준 최초/최종 등장 시점 집계
    for frame_info in ocr_results:
        frame_idx = frame_info.frame_idx
        frame_time = frame_info.timestamp_sec
        for track_id in frame_info.track_ids:
            if track_id not in first_frame_by_track:
                first_frame_by_track[track_id] = frame_idx
                first_time_by_track[track_id] = frame_time
            last_frame_by_track[track_id] = frame_idx

    # Union-Find 초기화: 처음에는 각 트랙이 자기 자신을 루트로 가짐
    parent: dict[int, int] = {track_id: track_id for track_id in last_frame_by_track}

    def find_root(x: int) -> int:
        # 루트를 찾을 때 경로 압축(path compression) 적용
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union_track(a: int, b: int) -> None:
        # a, b가 속한 그룹의 루트를 찾아 하나로 병합
        ra = find_root(a)
        rb = find_root(b)
        if ra == rb:
            return
        # 더 먼저 등장한 트랙을 루트로 유지(동일하면 id 작은 쪽)
        a_first = first_frame_by_track.get(ra, 10**12)
        b_first = first_frame_by_track.get(rb, 10**12)
        if (a_first, ra) <= (b_first, rb):
            parent[rb] = ra
        else:
            parent[ra] = rb

    # 완전 동시 종료가 아니어도 근접 종료(기본 6프레임) 허용
    end_frame_tolerance = 6
    start_time_tolerance_sec = 2.0
    tracks_sorted_by_last_frame = sorted(last_frame_by_track.items(), key=lambda item: item[1])
    track_count = len(tracks_sorted_by_last_frame)
    for i in range(track_count - 1):
        # 기준 트랙 A와 종료 시점이 가까운 다음 트랙들 비교
        track_id_a, last_frame_a = tracks_sorted_by_last_frame[i]

        j = i + 1
        while j < track_count:
            track_id_b, last_frame_b = tracks_sorted_by_last_frame[j]
            # 종료 프레임 차이가 허용치를 넘으면 이후 트랙은 비교 제외
            if last_frame_b - last_frame_a > end_frame_tolerance:
                break
            
            first_time_a = first_time_by_track.get(track_id_a)
            first_time_b = first_time_by_track.get(track_id_b)
            if first_time_a is not None and first_time_b is not None:
                earlier_time, later_time = sorted((first_time_a, first_time_b))
                if later_time - earlier_time <= start_time_tolerance_sec:
                    # 두 트랙의 최초 등장 시각 차이가 2초 이내면 병합
                    union_track(track_id_a, track_id_b)

            j += 1

    # 각 프레임의 track_id를 최종 루트 id로 변환하여 병합 결과 반영
    for frame_info in ocr_results:
        merged_track_ids = []
        for tid in frame_info.track_ids:
            if tid in parent:
                merged_track_ids.append(find_root(tid))
            else:
                merged_track_ids.append(tid)
        frame_info.track_ids = merged_track_ids

    # 처리 로직 주석
    if visualize:
        viz_output_dir = jsonl_path_obj.parent / f"{vd_file.stem}_viz_frames"
        _dump_visualized_frames(vd_file, ocr_results, viz_output_dir)

    segments: list[Segment] = []
    track_frame_counts: dict[int, int] = defaultdict(int)
    track_text_counts: dict[int, Counter[str]] = defaultdict(Counter)
    track_start: dict[int, float] = {}
    track_end: dict[int, float] = {}

    # 프레임 정보 기반으로 카운트 집계
    for frame_info in ocr_results:
        track_ids = frame_info.track_ids
        rec_texts = [item.text for item in frame_info.spotting_items]
        rec_boxes = [item.bbox for item in frame_info.spotting_items]
        frame_time = frame_info.timestamp_sec

        # 현재 프레임 OCR 결과를 track_id별로 수집
        frame_track_items: dict[int, list[tuple[float, float, float, float, str]]] = defaultdict(list)
        for track_id, box, text in zip(track_ids, rec_boxes, rec_texts):
            x1, y1, x2, y2 = box
            frame_track_items[track_id].append((float(x1), float(y1), float(x2), float(y2), text))

        # 박스 배치(가로/세로)에 맞춰 정렬 후 프레임 텍스트 병합
        merged_text_by_track: dict[int, str] = {}
        for track_id, items in frame_track_items.items():
            total_width = sum(abs(x2 - x1) for x1, _, x2, _, _ in items)
            total_height = sum(abs(y2 - y1) for _, y1, _, y2, _ in items)
            is_horizontal_layout = total_width >= total_height

            if is_horizontal_layout:
                # 가로형: 위->아래, 같은 줄에서는 왼쪽->오른쪽
                items.sort(key=lambda it: ((it[1] + it[3]) / 2.0, (it[0] + it[2]) / 2.0))
            else:
                # 세로형: 오른쪽->왼쪽, 같은 열에서는 위->아래
                items.sort(key=lambda it: (-(it[0] + it[2]) / 2.0, (it[1] + it[3]) / 2.0))

            merged_text_by_track[track_id] = "\n".join(text for *_, text in items if text)

        # 병합된 프레임 텍스트를 트랙별 카운트/시간 범위로 누적
        for track_id, merged_text in merged_text_by_track.items():
            track_frame_counts[track_id] += 1
            track_text_counts[track_id][merged_text] += 1

            if track_id not in track_start or frame_time < track_start[track_id]:
                track_start[track_id] = frame_time
            if track_id not in track_end or frame_time > track_end[track_id]:
                track_end[track_id] = frame_time

    # 카운트 정보 기반으로 세그먼트 생성
    for track_id, frame_count in track_frame_counts.items():
        if frame_count < 8:
            continue
        text_counter = track_text_counts.get(track_id)
        if not text_counter:
            continue
    
        seg_text = text_counter.most_common(1)[0][0]
        segments.append(
            Segment(
                index=0,
                start=float(track_start[track_id]),
                end=float(track_end[track_id]),
                text=seg_text,
            )
        )

    # 처리 로직 주석
    segments.sort(key=lambda seg: (seg.start, seg.end, seg.text))

    # 다음 자막과 겹치지 않도록 현재 end 시간 조정
    min_subtitle_gap = 0.08  # 자막간 최소 간격 (약 2프레임)
    for idx in range(len(segments) - 1):
        current = segments[idx]
        next_seg = segments[idx + 1]
        latest_allowed_end = next_seg.start - min_subtitle_gap
        if current.end > latest_allowed_end:
            current.end = latest_allowed_end

    # 보정 과정에서 end < start 인 세그먼트 제거
    segments = [seg for seg in segments if seg.end >= seg.start]
    for i, seg in enumerate(segments, start=1):
        seg.index = i

    # 자막 타임스탬프 생성
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

    # 언어 감지 및 자막 경로 결정
    langs: List[str] = []
    sample_size = min(100, len(segments))
    for seg in random.sample(segments, sample_size):
        try:
            langs.append(detect(seg.text))
        except LangDetectException:
            continue
    most_common_lang = Counter(langs).most_common(1)[0][0] if langs else "un"
    srt_path = jsonl_path_obj.with_suffix(f".{most_common_lang}.srt")

    # SRT 파일 쓰기
    with srt_path.open("w", encoding="utf-8", newline="\n") as handle:
        first = True
        for seg in segments:
            if not first:
                handle.write("\n")
            first = False
            handle.write(f"{seg.index}\n")
            handle.write(f"{to_srt_timestamp(seg.start)} --> {to_srt_timestamp(seg.end)}\n")
            handle.write(f"{seg.text}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert OCR JSONL (frame_number,time,spotting_items) to SRT by merging contiguous "
            "identical or similar texts."
        )
    )
    parser.add_argument("jsonl", type=Path, help="Input JSONL path")
    parser.add_argument("--visualize", action="store_true", help="Dump frame visualization images")

    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    out = jsonl_to_srt(args.jsonl, visualize=args.visualize)
    print(out)


if __name__ == "__main__":
    main()
