import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract


try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


COCO_VEHICLE_CLASS_TO_TYPE = {
    # COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
    2: "4W",
    3: "2W",
    5: "HEAVY",
    7: "HEAVY",
    1: "2W",  # bicycle - treat as 2W
}

DEFAULT_VEHICLE_MODEL = os.environ.get("PARKING_VEHICLE_YOLO_MODEL", "yolov8n.pt")


@dataclass
class GateDetectionResult:
    vehicle_number: str
    plate_confidence: float
    vehicle_type: str
    vehicle_confidence: float
    frames_processed: int


def _normalize_plate(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return cleaned[:12]


def _looks_like_plate(s: str) -> bool:
    if not s:
        return False
    if len(s) < 6 or len(s) > 12:
        return False
    digits = sum(c.isdigit() for c in s)
    alphas = sum(c.isalpha() for c in s)
    # Relaxed rule: at least 2 letters and 2 digits.
    if digits < 2 or alphas < 2:
        return False
    return True


def _extract_plate_candidate(text: str) -> str:
    normalized = _normalize_plate(text)
    patterns = [
        r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{3,4}",
        r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}",
    ]
    for pat in patterns:
        m = re.search(pat, normalized)
        if m:
            return m.group(0)
    return normalized


def _generate_ocr_variants(crop_bgr: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    th_inv = cv2.bitwise_not(th_otsu)
    return [gray, th_otsu, th_adapt, th_inv]


def _ocr_plate_quick(crop_bgr: np.ndarray) -> Tuple[str, float]:
    """Single fast Tesseract pass (no image_to_data); good enough to short-circuit heavy OCR."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 12 or w < 24:
        return "", 0.0
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    raw = (pytesseract.image_to_string(th, config=cfg) or "").strip()
    plate = _extract_plate_candidate(raw)
    if not _looks_like_plate(plate):
        return "", 0.0
    # Synthetic confidence: quick path skips per-char scores.
    return plate, 72.0


def _ocr_plate_from_crop(
    crop_bgr: np.ndarray,
    quick_first: bool = True,
    stop_conf: float = 88.0,
) -> Tuple[str, float]:
    if quick_first:
        q_plate, q_conf = _ocr_plate_quick(crop_bgr)
        if q_plate:
            return q_plate, q_conf

    best_plate = ""
    best_conf = 0.0
    configs = [
        "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ]

    for variant in _generate_ocr_variants(crop_bgr):
        for config in configs:
            data = pytesseract.image_to_data(
                variant, config=config, output_type=pytesseract.Output.DICT
            )
            words: List[Tuple[str, float]] = []
            for txt, conf in zip(data.get("text", []), data.get("conf", [])):
                txt = (txt or "").strip()
                if not txt:
                    continue
                try:
                    c = float(conf)
                except Exception:
                    c = -1.0
                words.append((txt, c))

            if not words:
                continue
            raw_text = "".join(w for w, _ in words)
            plate = _extract_plate_candidate(raw_text)
            if not _looks_like_plate(plate):
                continue

            confidences = [c for _, c in words if c >= 0]
            conf_score = float(np.mean(confidences)) if confidences else 0.0
            conf_score = max(0.0, min(conf_score, 100.0))
            if conf_score >= best_conf:
                best_conf = conf_score
                best_plate = plate
                if best_conf >= stop_conf:
                    return best_plate, best_conf

    return best_plate, best_conf


def _find_plate_candidates(frame_bgr: np.ndarray, fast: bool = True) -> List[np.ndarray]:
    # Heuristic plate candidate detection (no dedicated plate model).
    # Works best when the plate is readable and fairly frontal.
    h, w = frame_bgr.shape[:2]
    if h < 80 or w < 80:
        return []

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if fast:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[np.ndarray] = []

    frame_area = float(h * w)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue

        aspect = bw / float(bh)
        area = bw * bh

        # Typical license plate is wider than tall
        if aspect < 1.2 or aspect > 8.5:
            continue
        # Filter tiny/noisy contours and overly large regions (slightly relaxed)
        if area < 0.0005 * frame_area or area > 0.12 * frame_area:
            continue

        # Crop with padding
        pad_x = int(bw * 0.15)
        pad_y = int(bh * 0.25)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        if (x2 - x1) < 30 or (y2 - y1) < 20:
            continue

        crop = frame_bgr[y1:y2, x1:x2]
        candidates.append(crop)

    # Deduplicate by approximate size (keep the best-looking crops)
    unique: List[np.ndarray] = []
    seen = set()
    for c in candidates:
        ch, cw = c.shape[:2]
        key = (cw // 10, ch // 10)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


def _crop_vehicle_plate_zone(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return frame_bgr

    vehicle = frame_bgr[y1:y2, x1:x2]
    vh, vw = vehicle.shape[:2]
    # Plate usually lies in lower-middle band of the vehicle.
    lx1 = int(vw * 0.15)
    lx2 = int(vw * 0.85)
    ly1 = int(vh * 0.45)
    ly2 = int(vh * 0.95)
    if lx2 <= lx1 or ly2 <= ly1:
        return vehicle
    return vehicle[ly1:ly2, lx1:lx2]


_VEHICLE_MODEL_CACHE: Dict[str, "YOLO"] = {}


def _detect_vehicle_type_yolo(
    model: "YOLO", frame_bgr: np.ndarray, imgsz: int = 480
) -> Tuple[str, float, Optional[Tuple[int, int, int, int]]]:
    # Ultralytics expects RGB or BGR arrays; it handles conversion internally.
    # We run CPU because user asked for CPU-only.
    results = model.predict(
        source=frame_bgr,
        device="cpu",
        imgsz=imgsz,
        classes=list(COCO_VEHICLE_CLASS_TO_TYPE.keys()),
        verbose=False,
    )
    r = results[0]
    if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
        return "4W", 0.0, None

    best_type = "4W"
    best_conf = 0.0
    best_bbox: Optional[Tuple[int, int, int, int]] = None

    for box in r.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id not in COCO_VEHICLE_CLASS_TO_TYPE:
            continue
        vtype = COCO_VEHICLE_CLASS_TO_TYPE[cls_id]
        if conf > best_conf:
            best_type = vtype
            best_conf = conf
            xyxy = box.xyxy[0].tolist()
            best_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
    return best_type, best_conf, best_bbox


def _instant_frame_indices(cap: cv2.VideoCapture, max_samples: int = 3) -> List[int]:
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        return []
    if n <= max_samples:
        return list(range(n))
    return sorted({0, max(0, n // 2), n - 1})[:max_samples]


def _detect_from_offline_video_instant(
    video_path: str,
    resize_width: int = 480,
    max_contour_crops: int = 2,
) -> GateDetectionResult:
    """
    Minimal work: 1–3 frames, no YOLO, stop on first good plate.
    Vehicle type defaults to 4W (user can correct in UI).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for reading.")

    frames: List[np.ndarray] = []
    try:
        idxs = _instant_frame_indices(cap, 3)
        if idxs:
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)
        if not frames:
            for _ in range(3):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames.append(frame)
    finally:
        cap.release()

    plate_freq: Counter = Counter()
    plate_conf_scores: Dict[str, List[float]] = defaultdict(list)
    processed = 0

    for frame in frames:
        processed += 1
        h, w = frame.shape[:2]
        if resize_width > 0 and w > resize_width:
            scale = resize_width / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        plate, pconf = _ocr_plate_from_crop(frame, quick_first=True, stop_conf=88.0)
        if plate:
            plate_freq[plate] += 1
            plate_conf_scores[plate].append(pconf)
            break

        for crop in _find_plate_candidates(frame, fast=True)[:max_contour_crops]:
            plate, pconf = _ocr_plate_from_crop(crop, quick_first=True, stop_conf=88.0)
            if plate:
                plate_freq[plate] += 1
                plate_conf_scores[plate].append(pconf)
                break
        if plate_freq:
            break

    if not plate_freq:
        return GateDetectionResult(
            vehicle_number="",
            plate_confidence=0.0,
            vehicle_type="4W",
            vehicle_confidence=0.0,
            frames_processed=processed,
        )

    best_plate = plate_freq.most_common(1)[0][0]
    plate_conf = float(np.mean(plate_conf_scores[best_plate])) if plate_conf_scores.get(best_plate) else 0.0
    return GateDetectionResult(
        vehicle_number=best_plate,
        plate_confidence=plate_conf,
        vehicle_type="4W",
        vehicle_confidence=0.0,
        frames_processed=processed,
    )


def detect_from_offline_video(
    video_path: str,
    max_frames: int = 60,
    frame_stride: int = 12,
    resize_width: int = 640,
    yolo_imgsz: int = 480,
    max_candidates_per_frame: int = 8,
    early_stop_plate_votes: int = 4,
    early_stop_min_avg_conf: float = 55.0,
    early_stop_min_frames: int = 4,
    instant: bool = False,
) -> GateDetectionResult:
    """
    Offline CPU gate processing:
    - sample frames
    - infer vehicle type via YOLO (if ultralytics is available)
    - infer license plate via contour heuristics + Tesseract OCR
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    if instant:
        return _detect_from_offline_video_instant(video_path)

    yolo_model = None
    if YOLO is not None:
        global _VEHICLE_MODEL_CACHE
        if DEFAULT_VEHICLE_MODEL not in _VEHICLE_MODEL_CACHE:
            # This may download weights on first run.
            _VEHICLE_MODEL_CACHE[DEFAULT_VEHICLE_MODEL] = YOLO(DEFAULT_VEHICLE_MODEL)
        yolo_model = _VEHICLE_MODEL_CACHE[DEFAULT_VEHICLE_MODEL]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for reading.")

    frame_count = 0
    processed = 0
    plate_conf_scores: Dict[str, List[float]] = defaultdict(list)
    plate_freq: Counter = Counter()

    vehicle_type_votes: Counter = Counter()
    vehicle_conf_scores: List[float] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                break

            if frame_count % max(1, frame_stride) != 0:
                frame_count += 1
                continue

            frame_count += 1
            processed += 1
            if processed > max_frames:
                break

            h, w = frame.shape[:2]
            if resize_width > 0 and w > resize_width:
                scale = resize_width / float(w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # Vehicle type detection (best-effort).
            vehicle_bbox = None
            if yolo_model is not None:
                vtype, vconf, vehicle_bbox = _detect_vehicle_type_yolo(
                    yolo_model, frame, imgsz=yolo_imgsz
                )
                vehicle_type_votes[vtype] += 1
                if vconf > 0:
                    vehicle_conf_scores.append(vconf)
            else:
                # If YOLO isn't available, keep a default so billing still works.
                vehicle_type_votes["4W"] += 1

            # License plate detection + OCR (ROI-first; full frame only if needed).
            candidates: List[np.ndarray] = []
            if vehicle_bbox is not None:
                roi = _crop_vehicle_plate_zone(frame, vehicle_bbox)
                candidates.extend(_find_plate_candidates(roi, fast=True))
                candidates.append(roi)
            best_plate_this_frame = ""
            best_plate_conf = 0.0
            # Stop scanning crops once quick OCR finds a plausible plate (avoids heavy Tesseract per crop).
            inner_good_conf = 70.0

            for crop in candidates[:max_candidates_per_frame]:
                plate, pconf = _ocr_plate_from_crop(crop, quick_first=True, stop_conf=88.0)
                if plate and pconf >= best_plate_conf:
                    best_plate_this_frame = plate
                    best_plate_conf = pconf
                    if best_plate_conf >= inner_good_conf:
                        break

            if not best_plate_this_frame and vehicle_bbox is None:
                candidates2 = _find_plate_candidates(frame, fast=True)
                candidates2.append(frame)
                for crop in candidates2[:max_candidates_per_frame]:
                    plate, pconf = _ocr_plate_from_crop(
                        crop, quick_first=True, stop_conf=88.0
                    )
                    if plate and pconf >= best_plate_conf:
                        best_plate_this_frame = plate
                        best_plate_conf = pconf
                        if best_plate_conf >= inner_good_conf:
                            break

            if not best_plate_this_frame and vehicle_bbox is not None:
                for crop in _find_plate_candidates(frame, fast=True)[: max(0, max_candidates_per_frame - 2)]:
                    plate, pconf = _ocr_plate_from_crop(
                        crop, quick_first=True, stop_conf=88.0
                    )
                    if plate and pconf >= best_plate_conf:
                        best_plate_this_frame = plate
                        best_plate_conf = pconf
                        if best_plate_conf >= inner_good_conf:
                            break
                if not best_plate_this_frame:
                    plate, pconf = _ocr_plate_from_crop(
                        frame, quick_first=True, stop_conf=88.0
                    )
                    if plate and pconf >= best_plate_conf:
                        best_plate_this_frame = plate
                        best_plate_conf = pconf

            if best_plate_this_frame:
                plate_freq[best_plate_this_frame] += 1
                plate_conf_scores[best_plate_this_frame].append(best_plate_conf)

            if plate_freq and processed >= early_stop_min_frames:
                top_plate, top_votes = plate_freq.most_common(1)[0]
                if top_votes >= early_stop_plate_votes:
                    confs = plate_conf_scores.get(top_plate, [])
                    avg_c = float(np.mean(confs)) if confs else 0.0
                    if avg_c >= early_stop_min_avg_conf:
                        break
    finally:
        cap.release()

    # Choose best plate across frames.
    if not plate_freq:
        vehicle_type = (
            vehicle_type_votes.most_common(1)[0][0] if vehicle_type_votes else "4W"
        )
        vehicle_conf = float(np.mean(vehicle_conf_scores)) if vehicle_conf_scores else 0.0
        return GateDetectionResult(
            vehicle_number="",
            plate_confidence=0.0,
            vehicle_type=vehicle_type,
            vehicle_confidence=vehicle_conf,
            frames_processed=processed,
        )

    best_plate = ""
    best_score = -1.0
    for plate, freq in plate_freq.items():
        conf_list = plate_conf_scores.get(plate, [])
        avg_conf = float(np.mean(conf_list)) if conf_list else 0.0
        # Combine frequency and confidence. Frequency helps stabilize OCR.
        score = avg_conf * 0.7 + freq * 10.0
        if score > best_score:
            best_score = score
            best_plate = plate

    plate_conf = float(np.mean(plate_conf_scores[best_plate])) if plate_conf_scores.get(best_plate) else 0.0
    vehicle_type = vehicle_type_votes.most_common(1)[0][0] if vehicle_type_votes else "4W"
    vehicle_conf = float(np.mean(vehicle_conf_scores)) if vehicle_conf_scores else 0.0

    return GateDetectionResult(
        vehicle_number=best_plate,
        plate_confidence=plate_conf,
        vehicle_type=vehicle_type,
        vehicle_confidence=vehicle_conf,
        frames_processed=processed,
    )

