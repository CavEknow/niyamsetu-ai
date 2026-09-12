"""
app.py — NiyamSetu AI v2
Main Flask entry point with QR/Barcode scanning support.

Run: python app.py
"""

import os
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"]          = os.getenv("FLASK_SECRET_KEY", "change-me")
app.config["MAX_CONTENT_LENGTH"]  = 16 * 1024 * 1024   # 16 MB
app.config["UPLOAD_FOLDER"]       = os.getenv("UPLOAD_FOLDER", "uploads")

CORS(app)

# ─ JWT ─
from auth import init_jwt, auth_bp, admin_bp
jwt = init_jwt(app)

# ─ Blueprints ─
app.register_blueprint(auth_bp)    # /api/auth/*
app.register_blueprint(admin_bp)   # /api/admin/*

from qr_routes import qr_bp
app.register_blueprint(qr_bp)      # /api/qr/* + /qr page

# ─────────────────────── Page routes ───────────────────────

@app.route("/")
@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/inspect")
def inspect_page():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")

# ─────────────────────── API routes ───────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "app": "NiyamSetu AI", "version": "2.0.0"})

@app.route("/api/dashboard/stats")
def dashboard_stats():
    from flask_jwt_extended import jwt_required, verify_jwt_in_request
    try:
        verify_jwt_in_request()
    except Exception:
        return jsonify({"error": "Authentication required."}), 401
    from database import dashboard_stats as get_stats
    return jsonify(get_stats()), 200

@app.route("/api/inspections")
def list_inspections():
    from flask_jwt_extended import jwt_required, verify_jwt_in_request, get_jwt_identity, get_jwt
    try:
        verify_jwt_in_request()
    except Exception:
        return jsonify({"error": "Authentication required."}), 401
    from database import inspection_list
    claims = get_jwt()
    role   = claims.get("role", "viewer")
    uid    = get_jwt_identity()
    inspector_id = None if role == "admin" else uid
    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 20))
    return jsonify(inspection_list(inspector_id=inspector_id, page=page, limit=limit)), 200

# ─────────────────────── Error handlers ───────────────────────

@app.errorhandler(404)
def not_found(_):    return jsonify({"error": "Not found."}), 404

@app.errorhandler(413)
def too_large(_):    return jsonify({"error": "File too large. Max 16 MB."}), 413

@app.errorhandler(500)
def server_error(e): return jsonify({"error": "Server error.", "detail": str(e)}), 500

# ─────────────────────── Run ───────────────────────

if __name__ == "__main__":
    debug = os.getenv("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", port=5000, debug=debug)
