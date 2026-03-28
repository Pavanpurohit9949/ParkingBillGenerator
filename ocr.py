import re

import cv2
import numpy as np
import pytesseract


def normalize_plate(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return cleaned


def _looks_like_plate(s: str) -> bool:
    if not s:
        return False
    if len(s) < 6 or len(s) > 12:
        return False
    digits = sum(c.isdigit() for c in s)
    alphas = sum(c.isalpha() for c in s)
    return digits >= 2 and alphas >= 2


def _extract_plate_candidate(text: str) -> str:
    normalized = normalize_plate(text)
    patterns = [
        r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{3,4}",
        r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}",
    ]
    for pat in patterns:
        m = re.search(pat, normalized)
        if m:
            return m.group(0)
    return normalized


def extract_vehicle_number(image_bytes: bytes) -> str:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return ""

    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side > 1200:
        scale = 1200.0 / max_side
        image = cv2.resize(
            image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    quick = normalize_plate((pytesseract.image_to_string(otsu, config=cfg) or "").strip())
    plate = _extract_plate_candidate(quick)
    if _looks_like_plate(plate):
        return plate[:12]

    denoised = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    threshold = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    text = pytesseract.image_to_string(threshold, config="--psm 7")
    plate = normalize_plate(text)
    return plate[:12]
