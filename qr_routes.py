"""
qr_routes.py — NiyamSetu AI v2
Flask blueprint for QR code and barcode scanning API routes.

Register in app.py:
    from qr_routes import qr_bp
    app.register_blueprint(qr_bp)
"""

import os
import uuid
from pathlib import Path
from flask import Blueprint, request, jsonify, render_template, g
from flask_jwt_extended import jwt_required, get_jwt, get_jwt_identity

import database as db
from qr_scanner import run_qr_scan, scan_image, lookup_barcode
from auth import require_min_role, _client_ip

qr_bp = Blueprint("qr", __name__)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
ALLOWED_EXT   = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ─────────────────────────────────────────
#  Page routes
# ─────────────────────────────────────────

@qr_bp.route("/qr")
def qr_page():
    """Serve the QR scanner UI page."""
    return render_template("qr_scan.html")


# ─────────────────────────────────────────
#  API: scan image for QR/barcode
# ─────────────────────────────────────────

@qr_bp.route("/api/qr/scan", methods=["POST"])
@jwt_required()
@require_min_role("inspector")
def api_qr_scan():
    """
    POST /api/qr/scan
    Form-data:  image (file)
    Optional:   run_compliance=true (also run LM Rules 2011 check)
                lookup_online=true  (query Open Food Facts)

    Returns full QR scan result + optional compliance report.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    f = request.files["image"]
    if not f.filename or not _allowed(f.filename):
        return jsonify({"error": "Invalid file type. Upload JPG, PNG, or WEBP."}), 400

    # Save upload
    Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
    ext      = f.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4()}.{ext}"
    path     = os.path.join(UPLOAD_FOLDER, filename)
    f.save(path)

    run_compliance  = request.form.get("run_compliance",  "true").lower()  == "true"
    lookup_online   = request.form.get("lookup_online",   "true").lower()  == "true"
    inspector_id    = get_jwt_identity()

    # Run QR scan
    result = run_qr_scan(path, lookup_online=lookup_online)
    result["image_filename"] = filename

    # Optional: also run LM Rules compliance on extracted fields
    if run_compliance and result.get("found") and result.get("merged_fields"):
        try:
            from rules import run_compliance_check
            job_id     = result.get("codes", [{}])[0].get("data", filename)[:36]
            compliance = run_compliance_check(job_id, result["merged_fields"])
            result["compliance"] = compliance

            # Save inspection to MongoDB
            save_doc = {
                "job_id":          job_id,
                "source":          "qr_scan",
                "image_path":      path,
                "fields":          result["merged_fields"],
                "results":         compliance["results"],
                "summary":         compliance["summary"],
                "overall_status":  compliance["overall_status"],
                "qr_codes":        result["codes"],
                "off_data":        result.get("off_data"),
            }
            db.inspection_save(save_doc, inspector_id)
        except Exception as exc:
            result["compliance_error"] = str(exc)

    # Audit log
    db.audit_log(
        user_id=inspector_id,
        action="qr_scan",
        detail=f"Found={result['found']}, codes={len(result.get('codes', []))}",
        ip=_client_ip(),
    )

    return jsonify(result), 200


# ─────────────────────────────────────────
#  API: barcode lookup only (no image)
# ─────────────────────────────────────────

@qr_bp.route("/api/qr/lookup/<barcode>", methods=["GET"])
@jwt_required()
@require_min_role("viewer")
def api_barcode_lookup(barcode: str):
    """
    GET /api/qr/lookup/<barcode>
    Lookup a barcode number on Open Food Facts.
    Works without an image — useful for manual barcode entry.
    """
    data = lookup_barcode(barcode)
    if not data:
        return jsonify({"found": False, "message": "Barcode not found on Open Food Facts."}), 404
    return jsonify({"found": True, "product": data}), 200


# ─────────────────────────────────────────
#  API: recent QR scan history
# ─────────────────────────────────────────

@qr_bp.route("/api/qr/history", methods=["GET"])
@jwt_required()
@require_min_role("inspector")
def api_qr_history():
    """
    GET /api/qr/history?page=1&limit=20
    Returns past QR scan inspections for the current user.
    """
    claims       = get_jwt()
    role         = claims.get("role", "inspector")
    inspector_id = get_jwt_identity()
    page         = int(request.args.get("page",  1))
    limit        = int(request.args.get("limit", 20))

    # Admins see all; inspectors see only their own
    uid = None if role == "admin" else inspector_id
    return jsonify(db.inspection_list(
        inspector_id=uid,
        page=page,
        limit=limit,
    )), 200
