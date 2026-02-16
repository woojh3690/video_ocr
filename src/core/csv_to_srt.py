import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from typing import Iterable, List, Optional

import numpy as np
import unicodedata
from norfair import Detection, Tracker

from core.paddle_client import SpottingItem

@dataclass
class FrameInfo:
    frame_idx: int
    timestamp_sec: float
    spotting_items: List[SpottingItem]

@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str

def fine_video(jsonl_path_obj: Path) -> Path:
    for file in jsonl_path_obj.parent.iterdir():
        if file.name == jsonl_path_obj.name and file.suffix not in [".jsonl", ".srt"]:
            return file
    raise FileNotFoundError("Video file not found")

def jsonl_to_srt(jsonl_path: str, visualize=True) -> None:
        # 경로 확인
        jsonl_path_obj = Path(jsonl_path)
        if not jsonl_path_obj.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path_obj}")
        if visualize:
            vd_file = fine_video(jsonl_path_obj)
        
        # jsonl 파일 읽기
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
                ocr_results.append(
                    FrameInfo(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        spotting_items=spotting_items
                    )
                )

        # 트래킹 전, 글자 수(문자 카테고리 L 기준)가 1개 이하인 OCR 결과 제거
        for frame_info in ocr_results:
            rec_texts = frame_info.rec_texts
            if not rec_texts:
                continue

            keep_indices = []
            for idx, text in enumerate(rec_texts):
                letter_count = sum(1 for ch in text if unicodedata.category(ch).startswith("L"))
                if letter_count > 1:
                    keep_indices.append(idx)

            # 인덱스가 연동되는 OCR 필드들을 동일하게 필터링
            if len(keep_indices) != len(rec_texts):
                frame_info.rec_texts = [frame_info.rec_texts[idx] for idx in keep_indices]
                frame_info.rec_polys = [frame_info.rec_polys[idx] for idx in keep_indices]
                frame_info.rec_boxes = [frame_info.rec_boxes[idx] for idx in keep_indices]

        # 트래커 초기화
        text_weight_base = 0.24
        max_text_weight_at_conf = 0.90
        strong_mismatch_similarity = 0.35
        mismatch_penalty = 0.08

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

        def hybrid_distance(det: Detection, tracked_object) -> float:
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

        # 트래커를 사용하여 텍스트 박스별 track id를 계산
        for frame_info in ocr_results:
            rec_boxes = frame_info.rec_boxes
            rec_texts = frame_info.rec_texts
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

            # 트래킹
            frame_info.track_ids = [None] * len(rec_boxes)
            tracked_objects = tracker.update(detections=detections)

            # 이번 프레임 detection 객체 identity만 허용
            curr_det_ids = {id(d) for d in detections}

            for t in tracked_objects:
                if id(t.last_detection) not in curr_det_ids:
                    continue  # 이전 프레임 last_detection인 트랙 제외
                det_index = t.last_detection.data["det_index"]
                frame_info.track_ids[det_index] = t.id

            if any(track_id is None for track_id in frame_info.track_ids):
                raise RuntimeError(
                    f"Unmatched detection found at frame {frame_info.frame_idx}: {frame_info.track_ids}"
                )

        # 종료 시점이 가깝고, 시작 시점도 가까운 트랙은 같은 자막으로 간주
        first_frame_by_track: dict[int, int] = {}
        first_time_by_track: dict[int, float] = {}
        last_frame_by_track: dict[int, int] = {}

        # track_id 기준으로 최초/최종 등장 시점 집계
        for frame_info in ocr_results:
            frame_idx = frame_info.frame_idx
            frame_time = frame_info.timestamp_sec
            for track_id in frame_info.track_ids:
                if track_id not in first_frame_by_track:
                    first_frame_by_track[track_id] = frame_idx
                    first_time_by_track[track_id] = frame_time
                last_frame_by_track[track_id] = frame_idx

        # Union-Find 초기화: 처음에는 각 트랙이 자기 자신을 대표(root)로 가짐
        parent: dict[int, int] = {track_id: track_id for track_id in last_frame_by_track}

        def find_root(x: int) -> int:
            # 대표를 찾을 때 경로 압축(path compression)으로 다음 탐색을 빠르게 함
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union_track(a: int, b: int) -> None:
            # a, b가 속한 그룹의 대표를 찾아 하나의 그룹으로 병합
            ra = find_root(a)
            rb = find_root(b)
            if ra == rb:
                return
            # 더 먼저 등장한 트랙 쪽을 대표로 유지(동률이면 id 작은 쪽)
            a_first = first_frame_by_track.get(ra, 10**12)
            b_first = first_frame_by_track.get(rb, 10**12)
            if (a_first, ra) <= (b_first, rb):
                parent[rb] = ra
            else:
                parent[ra] = rb

        # 완전 동시 종료가 아니어도 근접 종료(±6프레임)는 허용.
        end_frame_tolerance = 6
        start_time_tolerance_sec = 2.0
        tracks_sorted_by_last_frame = sorted(last_frame_by_track.items(), key=lambda item: item[1])
        track_count = len(tracks_sorted_by_last_frame)
        for i in range(track_count - 1):
            # 기준 트랙 A를 하나 잡고, 끝난 시점이 가까운 뒤쪽 트랙들과만 비교
            track_id_a, last_frame_a = tracks_sorted_by_last_frame[i]

            j = i + 1
            while j < track_count:
                track_id_b, last_frame_b = tracks_sorted_by_last_frame[j]
                # 종료 프레임 차이가 허용치보다 커지면 이후 트랙도 전부 제외 가능
                if last_frame_b - last_frame_a > end_frame_tolerance:
                    break

                first_time_a = first_time_by_track.get(track_id_a)
                first_time_b = first_time_by_track.get(track_id_b)
                if first_time_a is not None and first_time_b is not None:
                    earlier_time, later_time = sorted((first_time_a, first_time_b))
                    if later_time - earlier_time <= start_time_tolerance_sec:
                        # 먼저 등장한 트랙과 비교 대상 트랙의 첫 등장 시간이 2초 이내면 병합
                        union_track(track_id_a, track_id_b)

                j += 1

        # 각 프레임의 track_id를 최종 대표 id로 치환해 병합 결과를 반영
        for frame_info in ocr_results:
            merged_track_ids = []
            for tid in frame_info.track_ids:
                if tid in parent:
                    merged_track_ids.append(find_root(tid))
                else:
                    merged_track_ids.append(tid)
            frame_info.track_ids = merged_track_ids
        
        segments: list[Segment] = []
        track_frame_counts: dict[int, int] = defaultdict(int)
        track_text_counts: dict[int, Counter[str]] = defaultdict(Counter)
        track_start: dict[int, float] = {}
        track_end: dict[int, float] = {}

        # 트레킹 정보를 기반으로 카운팅
        for frame_info in ocr_results:
            track_ids = frame_info.track_ids
            rec_texts = frame_info.rec_texts
            rec_boxes = frame_info.rec_boxes
            frame_time = frame_info.timestamp_sec

            # 현재 프레임의 OCR 결과를 track_id별로 모음
            frame_track_items: dict[int, list[tuple[float, float, float, float, str]]] = defaultdict(list)
            for track_id, box, text in zip(track_ids, rec_boxes, rec_texts):
                x1, y1, x2, y2 = box
                frame_track_items[track_id].append((float(x1), float(y1), float(x2), float(y2), text))

            # track_id별 박스 배치(가로/세로)에 맞춰 정렬 후 프레임 텍스트를 병합
            merged_text_by_track: dict[int, str] = {}
            for track_id, items in frame_track_items.items():
                total_width = sum(abs(x2 - x1) for x1, _, x2, _, _ in items)
                total_height = sum(abs(y2 - y1) for _, y1, _, y2, _ in items)
                is_horizontal_layout = total_width >= total_height

                if is_horizontal_layout:
                    # 가로형: 위 -> 아래, 같은 줄에서는 좌 -> 우
                    items.sort(key=lambda it: ((it[1] + it[3]) / 2.0, (it[0] + it[2]) / 2.0))
                else:
                    # 세로형: 우 -> 좌, 같은 열에서는 위 -> 아래
                    items.sort(key=lambda it: (-(it[0] + it[2]) / 2.0, (it[1] + it[3]) / 2.0))

                merged_text_by_track[track_id] = "\n".join(text for *_, text in items if text)

            # 병합된 프레임 텍스트를 트랙별 카운트/시간 범위에 누적
            for track_id, merged_text in merged_text_by_track.items():
                track_frame_counts[track_id] += 1
                track_text_counts[track_id][merged_text] += 1

                if track_id not in track_start or frame_time < track_start[track_id]:
                    track_start[track_id] = frame_time
                if track_id not in track_end or frame_time > track_end[track_id]:
                    track_end[track_id] = frame_time

        # 카운팅 정보를 기반으로 병합 진행
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

        # 자막 세그먼트 정렬
        segments.sort(key=lambda seg: (seg.start, seg.end, seg.text))

        # 자막이 뒤에 자막과 겹치지 않도록 seg.end 시간을 짧게 조정
        min_subtitle_gap = 0.08 # 자막간 최소 간격 (약 2프레임)
        for idx in range(len(segments) - 1):
            current = segments[idx]
            next_seg = segments[idx + 1]
            latest_allowed_end = next_seg.start - min_subtitle_gap
            if current.end > latest_allowed_end:
                current.end = latest_allowed_end

        # 보정 과정에서 end < start 로 역전된 자막은 제거
        segments = [seg for seg in segments if seg.end >= seg.start]
        for i, seg in enumerate(segments, start=1):
            seg.index = i
        
        # 자막 생성
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
        
        # srt 파일 초기화
        srt_path = jsonl_path_obj.with_suffix(".srt")
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
            "Convert OCR CSV (frame_number,time,text) to SRT by merging contiguous "
            "identical or similar texts."
        )
    )
    parser.add_argument("csv", type=Path, help="Input CSV path")

    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    out = jsonl_to_srt(args.csv)
    print(out)


if __name__ == "__main__":
    main()
