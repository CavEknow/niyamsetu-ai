"""
ocr.py — NiyamSetu AI
Dual OCR engine: Tesseract + EasyOCR with Hindi+English support,
preprocessing pipeline, and field extractor.

Requires:
    pip install pytesseract easyocr Pillow opencv-python-headless
    Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki
"""

import os
import re
import uuid
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("niyamsetu.ocr")

# ─────────────────────────────────────────
#  Tesseract setup
# ─────────────────────────────────────────

try:
    import pytesseract
    _tess_cmd = os.getenv("TESSERACT_CMD")
    if _tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tess_cmd
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    log.warning("pytesseract not installed — Tesseract OCR disabled.")

# ─────────────────────────────────────────
#  EasyOCR setup (lazy-loaded on first use)
# ─────────────────────────────────────────

_easy_reader = None

def _get_easy_reader():
    global _easy_reader
    if _easy_reader is None:
        try:
            import easyocr
            log.info("Loading EasyOCR model (first run may take a minute)...")
            _easy_reader = easyocr.Reader(["en", "hi"], gpu=False)
            log.info("EasyOCR ready.")
        except ImportError:
            log.warning("easyocr not installed — EasyOCR disabled.")
    return _easy_reader


# ─────────────────────────────────────────
#  Image preprocessing
# ─────────────────────────────────────────

def preprocess_image(image_path: str) -> tuple[np.ndarray, Image.Image]:
    """
    Load and preprocess the label image for best OCR accuracy.
    Returns (cv2_array, PIL_image).
    """
    # Load via OpenCV
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError(f"Cannot open image: {image_path}")

    # Resize to minimum 1200px wide for small labels
    h, w = img_cv.shape[:2]
    if w < 1200:
        scale  = 1200 / w
        img_cv = cv2.resize(img_cv, (1200, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold for uneven lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    # Sharpen
    kernel  = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)

    # PIL version (for Tesseract)
    pil_img = Image.fromarray(sharpened)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(2.0)

    return sharpened, pil_img


# ─────────────────────────────────────────
#  Individual OCR engines
# ─────────────────────────────────────────

def _run_tesseract(pil_img: Image.Image) -> str:
    """Run Tesseract with Hindi + English language packs."""
    if not TESSERACT_AVAILABLE:
        return ""
    lang = os.getenv("OCR_LANGUAGE", "eng+hin")
    config = "--oem 3 --psm 6"   # LSTM engine, assume uniform block of text
    try:
        text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
        log.info(f"Tesseract extracted {len(text)} chars.")
        return text
    except Exception as exc:
        log.warning(f"Tesseract failed: {exc}")
        return ""


def _run_easyocr(cv_img: np.ndarray) -> str:
    """Run EasyOCR and join all detected text blocks."""
    reader = _get_easy_reader()
    if reader is None:
        return ""
    try:
        results = reader.readtext(cv_img, detail=0, paragraph=True)
        text = "\n".join(results)
        log.info(f"EasyOCR extracted {len(text)} chars.")
        return text
    except Exception as exc:
        log.warning(f"EasyOCR failed: {exc}")
        return ""


# ─────────────────────────────────────────
#  Text merger
# ─────────────────────────────────────────

def _merge_texts(tess: str, easy: str) -> str:
    """
    Merge Tesseract and EasyOCR outputs.
    Strategy: use the longer one as base; append unique lines from the other.
    """
    base, extra = (tess, easy) if len(tess) >= len(easy) else (easy, tess)
    base_lines  = set(l.strip().lower() for l in base.splitlines() if l.strip())
    added_lines = [l for l in extra.splitlines() if l.strip()
                   and l.strip().lower() not in base_lines]
    merged = base
    if added_lines:
        merged += "\n" + "\n".join(added_lines)
    return merged


# ─────────────────────────────────────────
#  Field extractor
# ─────────────────────────────────────────

def extract_fields(raw_text: str) -> dict:
    """
    Parse raw OCR text into structured fields required by LM (PC) Rules 2011.
    Returns a dict with field keys and extracted string values (or None).
    """
    text = raw_text.upper()
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    def _find(patterns: list[str], flags=re.IGNORECASE) -> Optional[str]:
        for pat in patterns:
            m = re.search(pat, raw_text, flags)
            if m:
                return m.group(1).strip() if m.lastindex else m.group(0).strip()
        return None

    # — MRP —
    mrp = _find([
        r'MRP[:\s.]*(?:Rs\.?|INR|\u20b9)?\s*([\d,]+(?:\.\d{1,2})?)',
        r'(?:Rs\.?|\u20b9)\s*([\d,]+(?:\.\d{1,2})?)',
        r'MAXIMUM RETAIL PRICE[:\s]*(?:Rs\.?|\u20b9)?\s*([\d,]+(?:\.\d{1,2})?)',
    ])

    # — Net Quantity —
    net_qty = _find([
        r'NET\s+(?:WEIGHT|QTY|QUANTITY|CONTENT|WT)[:\s.]*([\d.]+\s*(?:g|gm|gms|kg|ml|l|ltr|litre|pcs|pieces|nos|tablets|tabs|units)?)',
        r'([\d.]+\s*(?:KG|G|GM|GMS|ML|L|LTR))(?:[\s,])',
        r'NET[:\s]*([\d.]+\s*(?:g|kg|ml|l|ltr))',
    ])

    # — Manufacturer —
    manufacturer = _find([
        r'(?:MANUFACTURED|MFD|MFG|MARKETED)\s*BY[:\s]*([A-Za-z0-9 &.,()-]{5,80})',
        r'(?:MANUFACTURER|MFGR)[:\s]*([A-Za-z0-9 &.,()-]{5,80})',
    ])

    # — Consumer Care —
    consumer_care = _find([
        r'(?:CONSUMER|CUSTOMER)\s*(?:CARE|HELPLINE|HELPDESK)[:\s]*([+\d\s-]{7,20})',
        r'(?:TOLL[- ]FREE|HELPLINE)[:\s]*([+\d\s-]{7,20})',
        r'(?:EMAIL|E-MAIL)[:\s]*([\w.+-]+@[\w.-]+\.[a-zA-Z]{2,})',
    ])

    # — Manufacture / Expiry Date —
    mfg_date = _find([
        r'(?:MFG|MFD|MANUFACTURED|MANUFACTURE)\s*(?:DATE)?[:\s.]*([A-Za-z]{3}[\s/-]?\d{2,4}|\d{2}[/-]\d{2,4})',
        r'(?:DOM|DATE OF MFG)[:\s]*([A-Za-z]{3}[\s/-]?\d{2,4}|\d{2}[/-]\d{2,4})',
    ])

    exp_date = _find([
        r'(?:EXP|EXPIRY|BEST BEFORE|USE BY|BB)[:\s.]*([A-Za-z]{3}[\s/-]?\d{2,4}|\d{2}[/-]\d{2,4})',
        r'(?:BEST BEFORE|USE BEFORE)[:\s]*([\d]{2}[/-][\d]{2,4})',
    ])

    # — Country of Origin —
    origin = _find([
        r'(?:COUNTRY OF ORIGIN|ORIGIN)[:\s]*([A-Za-z ]{3,30})',
        r'(?:MADE IN|PRODUCT OF)[:\s]*([A-Za-z ]{3,30})',
    ])

    # — FSSAI / License —
    fssai = _find([
        r'FSSAI[:\s#]*([\d]{14})',
        r'LIC\.?\s*NO\.?[:\s]*([\d]{14})',
    ])

    # — Commodity Name —
    # Use first non-empty line as product name (heuristic)
    commodity = lines[0] if lines else None

    return {
        "commodity":      commodity,
        "net_quantity":   net_qty,
        "mrp":            mrp,
        "manufacturer":   manufacturer,
        "consumer_care":  consumer_care,
        "mfg_date":       mfg_date,
        "exp_date":       exp_date,
        "origin":         origin,
        "fssai":          fssai,
        "raw_text":       raw_text,
    }


# ─────────────────────────────────────────
#  Main public function
# ─────────────────────────────────────────

def run_ocr(image_path: str, engine: str = None) -> dict:
    """
    Full OCR pipeline.

    Args:
        image_path: Path to the uploaded label image.
        engine:     'tesseract' | 'easyocr' | 'both' (default from .env / 'both')

    Returns:
        {
            job_id:     str,
            raw_text:   str,       # full merged OCR text
            fields:     dict,      # extracted label fields
            engines_used: list,
        }
    """
    engine = engine or os.getenv("OCR_ENGINE", "both")
    job_id = str(uuid.uuid4())

    log.info(f"[{job_id}] Starting OCR on: {image_path} (engine={engine})")

    try:
        cv_img, pil_img = preprocess_image(image_path)
    except Exception as exc:
        log.error(f"[{job_id}] Preprocessing failed: {exc}")
        return {"job_id": job_id, "error": str(exc), "raw_text": "", "fields": {}}

    tess_text  = ""
    easy_text  = ""
    engines_used = []

    if engine in ("tesseract", "both"):
        tess_text = _run_tesseract(pil_img)
        if tess_text:
            engines_used.append("tesseract")

    if engine in ("easyocr", "both"):
        easy_text = _run_easyocr(cv_img)
        if easy_text:
            engines_used.append("easyocr")

    raw_text = _merge_texts(tess_text, easy_text)

    if not raw_text.strip():
        log.warning(f"[{job_id}] No text extracted from image.")
        return {
            "job_id":      job_id,
            "raw_text":    "",
            "fields":      {},
            "engines_used": engines_used,
            "error":       "No text could be extracted. Check image quality.",
        }

    fields = extract_fields(raw_text)
    log.info(f"[{job_id}] Extraction done. Fields found: {[k for k,v in fields.items() if v and k != 'raw_text']}")

    return {
        "job_id":       job_id,
        "raw_text":     raw_text,
        "fields":       fields,
        "engines_used": engines_used,
    }
