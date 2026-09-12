"""
rules.py — NiyamSetu AI
Rule engine for Legal Metrology (Packaged Commodities) Rules, 2011.

Checks each mandatory declaration field and returns a structured
compliance report with PASS / FAIL / REVIEW status per rule.

Rule reference: LM (PC) Rules 2011, Rule 6 — Mandatory declarations.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────

@dataclass
class RuleResult:
    key:            str          # machine-readable rule identifier
    label:          str          # human-readable rule name
    rule_ref:       str          # LM (PC) Rules 2011 rule reference
    status:         str          # PASS | FAIL | REVIEW
    detected_value: Optional[str] = None   # what OCR found (or None)
    message:        str = ""     # explanation shown to officer
    severity:       str = "MEDIUM"  # HIGH | MEDIUM | LOW


@dataclass
class ComplianceReport:
    job_id:          str
    fields:          dict
    results:         list[RuleResult] = field(default_factory=list)
    summary:         dict            = field(default_factory=dict)
    overall_status:  str             = "PENDING"   # COMPLIANT | NON_COMPLIANT | NEEDS_REVIEW

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results"] = [asdict(r) for r in self.results]
        return d


# ─────────────────────────────────────────
#  Individual rule checkers
# ─────────────────────────────────────────

def _check_commodity(fields: dict) -> RuleResult:
    """Rule 6(1)(a): Name or generic name of the commodity."""
    val = (fields.get("commodity") or "").strip()
    if val and len(val) >= 3:
        return RuleResult(
            key="commodity", label="Name of Commodity",
            rule_ref="Rule 6(1)(a)",
            status="PASS", detected_value=val,
            message="Commodity name detected.",
            severity="HIGH",
        )
    return RuleResult(
        key="commodity", label="Name of Commodity",
        rule_ref="Rule 6(1)(a)",
        status="FAIL" if not val else "REVIEW",
        detected_value=val or None,
        message="Commodity name not clearly identified on label.",
        severity="HIGH",
    )


def _check_net_quantity(fields: dict) -> RuleResult:
    """Rule 6(1)(b): Net quantity in standard units."""
    val = (fields.get("net_quantity") or "").strip()
    valid_units = r'(?:g|gm|gms|kg|ml|l|ltr|litre|pcs|pieces|nos|tablets|tabs|units|mg)'
    if val and re.search(r'\d+\s*' + valid_units, val, re.IGNORECASE):
        return RuleResult(
            key="net_quantity", label="Net Quantity",
            rule_ref="Rule 6(1)(b)",
            status="PASS", detected_value=val,
            message="Net quantity with valid unit detected.",
            severity="HIGH",
        )
    if val:
        return RuleResult(
            key="net_quantity", label="Net Quantity",
            rule_ref="Rule 6(1)(b)",
            status="REVIEW", detected_value=val,
            message="Quantity detected but unit may be missing or non-standard. Manual review needed.",
            severity="HIGH",
        )
    return RuleResult(
        key="net_quantity", label="Net Quantity",
        rule_ref="Rule 6(1)(b)",
        status="FAIL", detected_value=None,
        message="Net quantity not found on label.",
        severity="HIGH",
    )


def _check_mrp(fields: dict) -> RuleResult:
    """Rule 6(1)(d): Maximum Retail Price (MRP) inclusive of all taxes."""
    val = (fields.get("mrp") or "").strip()
    if val and re.search(r'[\d]+', val):
        return RuleResult(
            key="mrp", label="MRP (Incl. All Taxes)",
            rule_ref="Rule 6(1)(d)",
            status="PASS", detected_value=f"\u20b9{val}",
            message="MRP declared on label.",
            severity="HIGH",
        )
    return RuleResult(
        key="mrp", label="MRP (Incl. All Taxes)",
        rule_ref="Rule 6(1)(d)",
        status="FAIL", detected_value=None,
        message="MRP not found. Must be printed as \u2018MRP (Incl. of all taxes) \u20b9 XX\u2019.",
        severity="HIGH",
    )


def _check_manufacturer(fields: dict) -> RuleResult:
    """Rule 6(1)(f): Name and address of manufacturer/packer/importer."""
    val = (fields.get("manufacturer") or "").strip()
    if val and len(val) >= 5:
        return RuleResult(
            key="manufacturer", label="Manufacturer Name & Address",
            rule_ref="Rule 6(1)(f)",
            status="PASS", detected_value=val,
            message="Manufacturer information detected.",
            severity="HIGH",
        )
    return RuleResult(
        key="manufacturer", label="Manufacturer Name & Address",
        rule_ref="Rule 6(1)(f)",
        status="FAIL", detected_value=val or None,
        message="Manufacturer name/address not found or too short.",
        severity="HIGH",
    )


def _check_consumer_care(fields: dict) -> RuleResult:
    """Rule 6(1): Consumer care contact (phone or email)."""
    val = (fields.get("consumer_care") or "").strip()
    has_phone = bool(val and re.search(r'\d{7,}', val))
    has_email = bool(val and re.search(r'@', val))
    if has_phone or has_email:
        return RuleResult(
            key="consumer_care", label="Consumer Care Contact",
            rule_ref="Rule 6(1)",
            status="PASS", detected_value=val,
            message="Consumer care contact detected.",
            severity="MEDIUM",
        )
    return RuleResult(
        key="consumer_care", label="Consumer Care Contact",
        rule_ref="Rule 6(1)",
        status="FAIL", detected_value=val or None,
        message="Consumer care phone/email not found.",
        severity="MEDIUM",
    )


def _check_mfg_date(fields: dict) -> RuleResult:
    """Rule 6(1)(e): Month and year of manufacture / packing."""
    val = (fields.get("mfg_date") or "").strip()
    if val and re.search(r'(?:\d{2}|[A-Za-z]{3})[/\s-]?\d{2,4}', val):
        return RuleResult(
            key="mfg_date", label="Month & Year of Manufacture",
            rule_ref="Rule 6(1)(e)",
            status="PASS", detected_value=val,
            message="Manufacturing date detected.",
            severity="MEDIUM",
        )
    if val:
        return RuleResult(
            key="mfg_date", label="Month & Year of Manufacture",
            rule_ref="Rule 6(1)(e)",
            status="REVIEW", detected_value=val,
            message="Partial date detected. Manual review required.",
            severity="MEDIUM",
        )
    return RuleResult(
        key="mfg_date", label="Month & Year of Manufacture",
        rule_ref="Rule 6(1)(e)",
        status="FAIL", detected_value=None,
        message="Manufacturing date not found on label.",
        severity="MEDIUM",
    )


def _check_origin(fields: dict) -> RuleResult:
    """Rule 6(1)(g): Country of origin for imported goods."""
    val = (fields.get("origin") or "").strip()
    if val and len(val) >= 2:
        return RuleResult(
            key="origin", label="Country of Origin",
            rule_ref="Rule 6(1)(g)",
            status="PASS", detected_value=val,
            message="Country of origin declared.",
            severity="HIGH",
        )
    return RuleResult(
        key="origin", label="Country of Origin",
        rule_ref="Rule 6(1)(g)",
        status="REVIEW", detected_value=None,
        message="Country of origin not detected. Required if product is imported.",
        severity="HIGH",
    )


def _check_fssai(fields: dict) -> RuleResult:
    """FSSAI License Number (mandatory for food products)."""
    val = (fields.get("fssai") or "").strip()
    if val and re.fullmatch(r'\d{14}', val):
        return RuleResult(
            key="fssai", label="FSSAI License No.",
            rule_ref="FSS Act 2006",
            status="PASS", detected_value=val,
            message="Valid 14-digit FSSAI license number detected.",
            severity="MEDIUM",
        )
    if val:
        return RuleResult(
            key="fssai", label="FSSAI License No.",
            rule_ref="FSS Act 2006",
            status="REVIEW", detected_value=val,
            message="FSSAI number detected but may be incomplete (need 14 digits).",
            severity="MEDIUM",
        )
    return RuleResult(
        key="fssai", label="FSSAI License No.",
        rule_ref="FSS Act 2006",
        status="REVIEW", detected_value=None,
        message="FSSAI number not detected. Required for food products.",
        severity="MEDIUM",
    )


# ─────────────────────────────────────────
#  Summary builder
# ─────────────────────────────────────────

def _build_summary(results: list[RuleResult]) -> tuple[dict, str]:
    total   = len(results)
    passes  = sum(1 for r in results if r.status == "PASS")
    fails   = sum(1 for r in results if r.status == "FAIL")
    reviews = sum(1 for r in results if r.status == "REVIEW")

    # Overall determination
    if fails == 0 and reviews == 0:
        overall = "COMPLIANT"
    elif fails > 0:
        overall = "NON_COMPLIANT"
    else:
        overall = "NEEDS_REVIEW"

    summary = {
        "total":    total,
        "passes":   passes,
        "fails":    fails,
        "reviews":  reviews,
        "status":   overall,
        "score":    round(passes / total * 100, 1) if total else 0,
    }
    return summary, overall


# ─────────────────────────────────────────
#  Main public function
# ─────────────────────────────────────────

def run_compliance_check(job_id: str, fields: dict) -> dict:
    """
    Run all LM (PC) Rules 2011 checks on the extracted OCR fields.

    Args:
        job_id:  Unique job identifier from OCR step.
        fields:  Dict returned by ocr.extract_fields().

    Returns:
        Full compliance report as a plain dict (ready to save to MongoDB).
    """
    checkers = [
        _check_commodity,
        _check_net_quantity,
        _check_mrp,
        _check_manufacturer,
        _check_consumer_care,
        _check_mfg_date,
        _check_origin,
        _check_fssai,
    ]

    results = [checker(fields) for checker in checkers]
    summary, overall = _build_summary(results)

    report = ComplianceReport(
        job_id=job_id,
        fields={k: v for k, v in fields.items() if k != "raw_text"},
        results=results,
        summary=summary,
        overall_status=overall,
    )
    return report.to_dict()


# ─────────────────────────────────────────
#  Convenience: run OCR + rules in one call
# ─────────────────────────────────────────

def inspect_label(image_path: str, inspector_id: str) -> dict:
    """
    Full pipeline: OCR → Rule Check → Save to DB.

    Usage in a Flask route:
        from rules import inspect_label
        result = inspect_label(saved_image_path, g.user_id)
        return jsonify(result)
    """
    from ocr import run_ocr
    import database as db

    # Step 1: OCR
    ocr_result = run_ocr(image_path)
    if ocr_result.get("error") and not ocr_result.get("fields"):
        return {"error": ocr_result["error"], "job_id": ocr_result.get("job_id")}

    # Step 2: Compliance check
    report = run_compliance_check(
        job_id=ocr_result["job_id"],
        fields=ocr_result["fields"],
    )
    report["raw_text"]     = ocr_result.get("raw_text", "")
    report["engines_used"] = ocr_result.get("engines_used", [])
    report["image_path"]   = image_path

    # Step 3: Save to MongoDB
    oid = db.inspection_save(report, inspector_id)
    report["_saved_id"] = oid

    return report
