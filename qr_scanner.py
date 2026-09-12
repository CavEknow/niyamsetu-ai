"""
qr_scanner.py — NiyamSetu AI v2
QR Code + Barcode scanner for packaged commodity label verification.

Features:
  - Scan QR codes and barcodes from uploaded images
  - Live webcam QR scanning (optional)
  - Open Food Facts API lookup by barcode (free, no key needed)
  - Extracts product name, MRP, manufacturer, net weight from QR data
  - Integrates with the existing rules.py compliance engine

Requires:
    pip install pyzbar opencv-python-headless Pillow requests
    Windows: also install Visual C++ Redistributable (for pyzbar)
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

log = logging.getLogger("niyamsetu.qr")

# ─────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────

# Open Food Facts free API (no key needed)
OFF_API = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"

# Supported barcode types
SUPPORTED_TYPES = [
    ZBarSymbol.QRCODE,
    ZBarSymbol.EAN13,
    ZBarSymbol.EAN8,
    ZBarSymbol.CODE128,
    ZBarSymbol.CODE39,
    ZBarSymbol.UPCA,
    ZBarSymbol.UPCE,
]


# ─────────────────────────────────────────
#  Image preprocessing for better QR detection
# ─────────────────────────────────────────

def _preprocess_for_qr(image_path: str) -> list[np.ndarray]:
    """
    Return multiple preprocessed versions of the image.
    Trying multiple versions dramatically improves QR detection on
    low-quality, skewed, or dark label images.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot open image: {image_path}")

    # Upscale small images
    h, w = img.shape[:2]
    if max(h, w) < 800:
        scale = 800 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)

    variants = []

    # 1. Original
    variants.append(img)

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants.append(gray)

    # 3. Sharpened
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    variants.append(sharp)

    # 4. Adaptive threshold (handles uneven lighting)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    variants.append(thresh)

    # 5. Inverted (some QR codes are dark-on-light vs light-on-dark)
    variants.append(cv2.bitwise_not(thresh))

    # 6. Denoised
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    variants.append(denoised)

    return variants


# ─────────────────────────────────────────
#  Core QR/barcode scanner
# ─────────────────────────────────────────

def scan_image(image_path: str) -> list[dict]:
    """
    Scan an image file for QR codes and barcodes.

    Returns a list of detected code dicts:
    [
        {
            "type":   "QRCODE" | "EAN13" | "CODE128" | ...
            "data":   "decoded string value",
            "raw":    bytes,
            "rect":   {"left": int, "top": int, "width": int, "height": int},
        }
    ]
    Returns empty list if nothing found.
    """
    seen_data: set = set()
    results: list[dict] = []

    try:
        variants = _preprocess_for_qr(image_path)
    except Exception as exc:
        log.error(f"QR preprocessing failed: {exc}")
        return []

    for variant in variants:
        try:
            codes = pyzbar.decode(variant, symbols=SUPPORTED_TYPES)
        except Exception:
            continue

        for code in codes:
            try:
                data = code.data.decode("utf-8", errors="replace").strip()
            except Exception:
                data = str(code.data)

            if data in seen_data:
                continue
            seen_data.add(data)

            rect = code.rect
            results.append({
                "type": code.type,
                "data": data,
                "raw":  code.data,
                "rect": {
                    "left":   rect.left,
                    "top":    rect.top,
                    "width":  rect.width,
                    "height": rect.height,
                },
            })
            log.info(f"QR/Barcode found: type={code.type}, data={data[:60]}")

        if results:
            break  # Stop trying variants once we have results

    return results


def scan_frame(frame: np.ndarray) -> list[dict]:
    """
    Scan a single webcam frame (numpy array from cv2.VideoCapture).
    Returns same format as scan_image().
    Suitable for real-time scanning in a loop.
    """
    seen_data: set = set()
    results: list[dict] = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for img in [frame, gray]:
        try:
            codes = pyzbar.decode(img, symbols=SUPPORTED_TYPES)
        except Exception:
            continue
        for code in codes:
            data = code.data.decode("utf-8", errors="replace").strip()
            if data in seen_data:
                continue
            seen_data.add(data)
            rect = code.rect
            results.append({
                "type": code.type,
                "data": data,
                "rect": {"left": rect.left, "top": rect.top,
                         "width": rect.width, "height": rect.height},
            })
    return results


# ─────────────────────────────────────────
#  Draw bounding boxes on image (for UI feedback)
# ─────────────────────────────────────────

def annotate_image(image_path: str, codes: list[dict], output_path: str) -> str:
    """
    Draw green bounding boxes around detected QR/barcodes and save.
    Returns the output_path.
    """
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    for code in codes:
        r = code["rect"]
        color = (0, 200, 80) if code["type"] == "QRCODE" else (255, 140, 0)
        cv2.rectangle(
            img,
            (r["left"], r["top"]),
            (r["left"] + r["width"], r["top"] + r["height"]),
            color, 3
        )
        label = f"{code['type']}: {code['data'][:30]}"
        cv2.putText(
            img, label,
            (r["left"], r["top"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    cv2.imwrite(output_path, img)
    return output_path


# ─────────────────────────────────────────
#  Open Food Facts barcode lookup
# ─────────────────────────────────────────

def lookup_barcode(barcode: str, timeout: int = 5) -> Optional[dict]:
    """
    Look up a barcode/EAN on Open Food Facts (free, no API key).
    Returns a cleaned product dict or None if not found.

    Returned dict keys (all may be None if unknown):
        product_name, brands, quantity, ingredients_text,
        nutriments, countries, image_url, off_url
    """
    if not REQUESTS_AVAILABLE:
        log.warning("requests not installed — barcode lookup disabled.")
        return None

    barcode = re.sub(r'[^0-9]', '', barcode)  # digits only
    if not barcode:
        return None

    url = OFF_API.format(barcode=barcode)
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": "NiyamSetu-AI/1.0"})
        data = resp.json()
    except Exception as exc:
        log.warning(f"Open Food Facts lookup failed: {exc}")
        return None

    if data.get("status") != 1:
        log.info(f"Barcode {barcode} not found on Open Food Facts.")
        return None

    p = data.get("product", {})
    return {
        "barcode":           barcode,
        "product_name":      p.get("product_name") or p.get("product_name_en"),
        "brands":            p.get("brands"),
        "quantity":          p.get("quantity"),
        "net_weight":        p.get("net_weight"),
        "ingredients_text":  p.get("ingredients_text"),
        "countries":         p.get("countries"),
        "categories":        p.get("categories"),
        "image_url":         p.get("image_front_url"),
        "off_url":           f"https://world.openfoodfacts.org/product/{barcode}",
        "nutriments":        p.get("nutriments", {}),
        "stores":            p.get("stores"),
        "packaging":         p.get("packaging"),
    }


# ─────────────────────────────────────────
#  QR data parser (for structured QR codes)
# ─────────────────────────────────────────

def parse_qr_data(raw: str) -> dict:
    """
    Parse the text content of a QR code into structured fields.
    Handles:
      - JSON QR codes: {"mrp": 150, "mfg": "Jan 2025", ...}
      - URL QR codes: https://example.com/product/12345
      - Key-value QR codes: MRP:150|MFG:Jan2025|NET:500g
      - Plain text (product name or barcode number)
    """
    parsed = {"raw": raw, "format": "unknown", "fields": {}}

    # 1. Try JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            parsed["format"] = "json"
            parsed["fields"] = {
                "mrp":          str(obj.get("mrp") or obj.get("MRP") or ""),
                "net_quantity": str(obj.get("net_qty") or obj.get("net_weight") or obj.get("quantity") or ""),
                "manufacturer": str(obj.get("manufacturer") or obj.get("brand") or ""),
                "mfg_date":     str(obj.get("mfg_date") or obj.get("mfg") or obj.get("manufacture_date") or ""),
                "exp_date":     str(obj.get("exp_date") or obj.get("expiry") or obj.get("best_before") or ""),
                "batch_no":     str(obj.get("batch") or obj.get("batch_no") or obj.get("lot") or ""),
                "origin":       str(obj.get("origin") or obj.get("country") or ""),
                "fssai":        str(obj.get("fssai") or obj.get("lic_no") or ""),
                "commodity":    str(obj.get("name") or obj.get("product") or obj.get("commodity") or ""),
            }
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Try URL
    if re.match(r'https?://', raw):
        parsed["format"] = "url"
        parsed["fields"]["url"] = raw
        # Try to extract product code from URL
        m = re.search(r'(?:product|item|sku|barcode)[/=]([\w-]+)', raw, re.I)
        if m:
            parsed["fields"]["product_code"] = m.group(1)
        return parsed

    # 3. Try key-value pairs (MRP:150|NET:500g or MRP=150;NET=500g)
    kv_matches = re.findall(
        r'(?:MRP|NET|NETQTY|MFG|EXP|BATCH|FSSAI|BRAND|ORIGIN)[:\s=]([^|;\n]+)',
        raw, re.IGNORECASE
    )
    if kv_matches:
        parsed["format"] = "key_value"
        for m in re.finditer(
            r'(MRP|NET|NETQTY|MFG|EXP|BATCH|FSSAI|BRAND|ORIGIN)[:\s=]([^|;\n]+)',
            raw, re.IGNORECASE
        ):
            key = m.group(1).lower()
            val = m.group(2).strip()
            key_map = {
                "mrp": "mrp", "net": "net_quantity", "netqty": "net_quantity",
                "mfg": "mfg_date", "exp": "exp_date", "batch": "batch_no",
                "fssai": "fssai", "brand": "manufacturer", "origin": "origin",
            }
            parsed["fields"][key_map.get(key, key)] = val
        return parsed

    # 4. Plain numeric barcode
    if re.fullmatch(r'[\d]{8,14}', raw.strip()):
        parsed["format"] = "barcode"
        parsed["fields"]["barcode"] = raw.strip()
        return parsed

    # 5. Plain text fallback
    parsed["format"] = "text"
    parsed["fields"]["commodity"] = raw[:200]
    return parsed


# ─────────────────────────────────────────
#  Main public function
# ─────────────────────────────────────────

def run_qr_scan(image_path: str, lookup_online: bool = True) -> dict:
    """
    Full QR/barcode pipeline for a label image.

    Args:
        image_path:     Path to uploaded label image.
        lookup_online:  If True, look up barcodes on Open Food Facts.

    Returns:
        {
            "found":        bool,
            "codes":        list of raw detected codes,
            "parsed":       list of parsed structured data per code,
            "merged_fields":dict of merged fields (for rules.py),
            "off_data":     dict from Open Food Facts (or None),
            "annotated_path": str path of annotated image (or None),
        }
    """
    log.info(f"Starting QR scan: {image_path}")

    # 1. Detect QR / barcodes
    codes = scan_image(image_path)

    if not codes:
        log.info("No QR codes or barcodes detected.")
        return {
            "found": False, "codes": [], "parsed": [],
            "merged_fields": {}, "off_data": None, "annotated_path": None,
        }

    # 2. Parse each detected code
    parsed_list = []
    for code in codes:
        p = parse_qr_data(code["data"])
        p["code_type"] = code["type"]
        parsed_list.append(p)

    # 3. Merge all extracted fields (first non-empty value wins)
    merged: dict = {}
    for p in parsed_list:
        for k, v in p["fields"].items():
            if v and k not in merged:
                merged[k] = v

    # 4. Open Food Facts lookup for barcodes
    off_data = None
    barcode_code = next(
        (c for c in codes if c["type"] in ("EAN13", "EAN8", "UPCA", "CODE128")),
        None
    )
    if barcode_code and lookup_online:
        barcode_num = re.sub(r'[^0-9]', '', barcode_code["data"])
        if barcode_num:
            off_data = lookup_barcode(barcode_num)
            if off_data:
                # Enrich merged fields from OFF data
                if not merged.get("commodity") and off_data.get("product_name"):
                    merged["commodity"] = off_data["product_name"]
                if not merged.get("manufacturer") and off_data.get("brands"):
                    merged["manufacturer"] = off_data["brands"]
                if not merged.get("net_quantity") and off_data.get("quantity"):
                    merged["net_quantity"] = off_data["quantity"]
                # Save to DB product master
                try:
                    import database as db
                    db.product_upsert(
                        barcode=barcode_num,
                        name=off_data.get("product_name") or "",
                        manufacturer=off_data.get("brands") or "",
                    )
                except Exception:
                    pass

    # 5. Annotate image
    ann_path = image_path.replace(".", "_qr_annotated.")
    try:
        annotate_image(image_path, codes, ann_path)
    except Exception:
        ann_path = None

    log.info(f"QR scan complete. Found {len(codes)} code(s). Fields: {list(merged.keys())}")

    return {
        "found":           True,
        "codes":           [{"type": c["type"], "data": c["data"]} for c in codes],
        "parsed":          parsed_list,
        "merged_fields":   merged,
        "off_data":        off_data,
        "annotated_path":  ann_path,
    }
