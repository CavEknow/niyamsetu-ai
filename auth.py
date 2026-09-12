"""
auth.py — NiyamSetu AI
JWT authentication, password hashing, role-based access control,
and all Flask auth route blueprints.

Requires:
    pip install flask flask-jwt-extended bcrypt python-dotenv

.env variables needed:
    JWT_SECRET_KEY=<strong-random-secret>
    JWT_ACCESS_TOKEN_EXPIRES=900          # seconds (default 15 min)
    JWT_REFRESH_TOKEN_EXPIRES=2592000     # seconds (default 30 days)
    MAX_LOGIN_ATTEMPTS=5
    LOCK_DURATION_MINUTES=30
"""

import os
import secrets
from datetime import datetime, timedelta
from functools import wraps

import bcrypt
from flask import Blueprint, request, jsonify, g
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    create_refresh_token,
    get_jwt_identity,
    get_jwt,
    jwt_required,
    decode_token,
)
from dotenv import load_dotenv

import database as db

load_dotenv()

# ─────────────────────────────────────────
#  Config constants
# ─────────────────────────────────────────

JWT_SECRET_KEY             = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
ACCESS_TOKEN_EXPIRES_SEC   = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES",  900))      # 15 min
REFRESH_TOKEN_EXPIRES_SEC  = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRES", 2592000))  # 30 days
MAX_LOGIN_ATTEMPTS         = int(os.getenv("MAX_LOGIN_ATTEMPTS", 5))
LOCK_DURATION_MINUTES      = int(os.getenv("LOCK_DURATION_MINUTES", 30))

# Valid roles in ascending privilege order
ROLE_HIERARCHY = ["viewer", "inspector", "admin"]


# ─────────────────────────────────────────
#  Flask-JWT-Extended bootstrap
# ─────────────────────────────────────────

def init_jwt(app):
    """
    Call this once in app.py after creating the Flask app:

        from auth import init_jwt
        init_jwt(app)
    """
    app.config["JWT_SECRET_KEY"]             = JWT_SECRET_KEY
    app.config["JWT_ACCESS_TOKEN_EXPIRES"]   = timedelta(seconds=ACCESS_TOKEN_EXPIRES_SEC)
    app.config["JWT_REFRESH_TOKEN_EXPIRES"]  = timedelta(seconds=REFRESH_TOKEN_EXPIRES_SEC)
    app.config["JWT_TOKEN_LOCATION"]         = ["headers", "cookies"]
    app.config["JWT_COOKIE_SECURE"]          = os.getenv("FLASK_ENV") == "production"
    app.config["JWT_COOKIE_SAMESITE"]        = "Lax"
    app.config["JWT_COOKIE_CSRF_PROTECT"]    = True

    jwt = JWTManager(app)

    # — Custom error responses —
    @jwt.expired_token_loader
    def expired_token(_header, _payload):
        return jsonify({"error": "Token has expired. Please sign in again."}), 401

    @jwt.invalid_token_loader
    def invalid_token(reason):
        return jsonify({"error": f"Invalid token: {reason}"}), 401

    @jwt.unauthorized_loader
    def missing_token(reason):
        return jsonify({"error": "Authentication required."}), 401

    @jwt.revoked_token_loader
    def revoked_token(_header, _payload):
        return jsonify({"error": "Token has been revoked. Please sign in again."}), 401

    return jwt


# ─────────────────────────────────────────
#  Password helpers
# ─────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*. Cost factor = 12 (good for 2024+)."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the stored bcrypt *hashed*."""
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def validate_password_strength(password: str) -> list[str]:
    """
    Returns a list of error strings. Empty list → password is strong enough.
    Policy: ≥10 chars, 1 uppercase, 1 digit, 1 special character.
    """
    errors = []
    if len(password) < 10:
        errors.append("Password must be at least 10 characters.")
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter.")
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit.")
    if not any(c in '!@#$%^&*()_+-=[]{};\':",.<>?/\\|`~' for c in password):
        errors.append("Password must contain at least one special character.")
    return errors


# ─────────────────────────────────────────
#  Token helpers
# ─────────────────────────────────────────

def _make_tokens(user: dict) -> dict:
    """
    Create a fresh access + refresh token pair for *user*.
    Persists the refresh token to MongoDB.
    """
    identity = str(user["_id"]) if "_id" in user else user["id"]
    additional_claims = {
        "role":       user["role"],
        "name":       user["name"],
        "email":      user["email"],
        "badge_id":   user.get("badge_id") or "",
        "department": user.get("department") or "",
    }

    access_token  = create_access_token(identity=identity, additional_claims=additional_claims)
    refresh_token = create_refresh_token(identity=identity, additional_claims=additional_claims)

    # Persist refresh token
    expires_at = db.utcnow() + timedelta(seconds=REFRESH_TOKEN_EXPIRES_SEC)
    db.refresh_token_save(refresh_token, identity, expires_at)

    return {
        "access_token":  access_token,
        "refresh_token": refresh_token,
        "token_type":    "Bearer",
        "expires_in":    ACCESS_TOKEN_EXPIRES_SEC,
        "role":          user["role"],
        "name":          user["name"],
        "email":         user["email"],
    }


# ─────────────────────────────────────────
#  RBAC decorators
# ─────────────────────────────────────────

def require_role(*roles):
    """
    Decorator: allow only users whose role is in *roles*.

    Usage:
        @app.route("/admin/users")
        @jwt_required()
        @require_role("admin")
        def list_users(): ...

        @app.route("/inspect")
        @jwt_required()
        @require_role("admin", "inspector")
        def inspect(): ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            claims = get_jwt()
            user_role = claims.get("role", "")
            if user_role not in roles:
                db.audit_log(
                    user_id=get_jwt_identity(),
                    action="forbidden",
                    detail=f"Role '{user_role}' tried to access {request.path}",
                    ip=_client_ip(),
                    success=False,
                )
                return jsonify({"error": "Access denied: insufficient permissions."}), 403
            # Expose claims to the view via Flask's g
            g.user_id   = get_jwt_identity()
            g.user_role = user_role
            g.claims    = claims
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def require_min_role(min_role: str):
    """
    Decorator: allow users whose role is >= *min_role* in ROLE_HIERARCHY.

    Usage:
        @require_min_role("inspector")   # allows inspector and admin
    """
    min_idx = ROLE_HIERARCHY.index(min_role)

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            claims    = get_jwt()
            user_role = claims.get("role", "viewer")
            user_idx  = ROLE_HIERARCHY.index(user_role) if user_role in ROLE_HIERARCHY else -1
            if user_idx < min_idx:
                db.audit_log(
                    user_id=get_jwt_identity(),
                    action="forbidden",
                    detail=f"Role '{user_role}' too low for {request.path}",
                    ip=_client_ip(),
                    success=False,
                )
                return jsonify({"error": "Access denied: insufficient permissions."}), 403
            g.user_id   = get_jwt_identity()
            g.user_role = user_role
            g.claims    = claims
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ─────────────────────────────────────────
#  Blueprint: /api/auth
# ─────────────────────────────────────────

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@auth_bp.route("/register", methods=["POST"])
def register():
    """
    POST /api/auth/register
    Body (JSON):
        name, email, password, role (optional), department, badge_id

    Returns 201 on success, 400/409 on error.
    NOTE: In production, restrict this to admin-only or invitation links.
    """
    data     = request.get_json(silent=True) or {}
    name     = (data.get("name") or "").strip()
    email    = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "")
    role     = (data.get("role") or "inspector").lower()
    dept     = (data.get("department") or "").strip()
    badge_id = (data.get("badge_id") or "").strip()

    # — Validation —
    if not name:
        return jsonify({"error": "Name is required."}), 400
    if not email or "@" not in email:
        return jsonify({"error": "A valid email address is required."}), 400
    if role not in ROLE_HIERARCHY:
        return jsonify({"error": f"Invalid role. Choose from: {', '.join(ROLE_HIERARCHY)}"}), 400

    strength_errors = validate_password_strength(password)
    if strength_errors:
        return jsonify({"error": "Weak password.", "details": strength_errors}), 400

    # — Create user —
    try:
        from pymongo.errors import DuplicateKeyError
        password_hash = hash_password(password)
        user_id = db.user_create(
            name=name,
            email=email,
            password_hash=password_hash,
            role=role,
            department=dept,
            badge_id=badge_id,
        )
        db.audit_log(
            user_id=user_id,
            action="register",
            detail=f"New {role} account: {email}",
            ip=_client_ip(),
        )
        return jsonify({"message": "Account created successfully.", "user_id": user_id}), 201

    except Exception as exc:
        if "duplicate" in str(exc).lower() or "11000" in str(exc):
            return jsonify({"error": "An account with this email already exists."}), 409
        return jsonify({"error": "Registration failed. Please try again."}), 500


@auth_bp.route("/login", methods=["POST"])
def login():
    """
    POST /api/auth/login
    Body (JSON): email, password, role (optional — verified against DB)

    Returns: access_token, refresh_token, role, name, email
    """
    data     = request.get_json(silent=True) or {}
    email    = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "")
    # role from body is informational; the authoritative role is from DB

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = db.user_get_by_email(email)

    # Unknown user — generic error to prevent user enumeration
    if not user:
        db.audit_log(None, "login_fail", f"Unknown email: {email}",
                     ip=_client_ip(), success=False)
        return jsonify({"error": "Invalid credentials."}), 401

    user_id = str(user["_id"])

    # Account disabled
    if not user.get("is_active", True):
        return jsonify({"error": "Your account has been deactivated. Contact your administrator."}), 403

    # Account locked?
    locked_until = user.get("locked_until")
    if locked_until and datetime.utcnow() < locked_until:
        remaining = int((locked_until - datetime.utcnow()).total_seconds() // 60) + 1
        return jsonify({
            "error": f"Account temporarily locked after too many failed attempts. "
                     f"Try again in {remaining} minute(s)."
        }), 429

    # Verify password
    if not verify_password(password, user["password_hash"]):
        attempts = db.user_record_failed_attempt(user_id)
        db.audit_log(user_id, "login_fail",
                     f"Bad password (attempt {attempts})",
                     ip=_client_ip(), success=False)
        if attempts >= MAX_LOGIN_ATTEMPTS:
            lock_until = datetime.utcnow() + timedelta(minutes=LOCK_DURATION_MINUTES)
            db.user_lock(user_id, lock_until)
            return jsonify({
                "error": f"Too many failed attempts. Account locked for {LOCK_DURATION_MINUTES} minutes."
            }), 429
        remaining_attempts = MAX_LOGIN_ATTEMPTS - attempts
        return jsonify({
            "error": f"Invalid credentials. {remaining_attempts} attempt(s) remaining."
        }), 401

    # — Success —
    db.user_update_login(user_id)
    tokens = _make_tokens(user)
    db.audit_log(user_id, "login", f"Role: {user['role']}", ip=_client_ip())
    return jsonify(tokens), 200


@auth_bp.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    """
    POST /api/auth/refresh
    Header: Authorization: Bearer <refresh_token>

    Issues a new access token (and rotates the refresh token).
    """
    old_refresh = request.headers.get("Authorization", "").split(" ")[-1]
    identity    = get_jwt_identity()

    # Verify refresh token is in DB (rotation guard)
    token_doc = db.refresh_token_get(old_refresh)
    if not token_doc:
        return jsonify({"error": "Refresh token not recognised or already used."}), 401

    # Delete old refresh token (rotation)
    db.refresh_token_delete(old_refresh)

    user = db.user_get_by_id(identity)
    if not user or not user.get("is_active", True):
        return jsonify({"error": "User not found or deactivated."}), 401

    tokens = _make_tokens(user)
    return jsonify(tokens), 200


@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    """
    POST /api/auth/logout
    Revokes the refresh token passed in the body (single device).
    Pass logout_all=true to invalidate all sessions.
    """
    data        = request.get_json(silent=True) or {}
    identity    = get_jwt_identity()
    logout_all  = data.get("logout_all", False)
    token       = data.get("refresh_token", "")

    if logout_all:
        db.refresh_tokens_delete_all_for_user(identity)
        db.audit_log(identity, "logout", "All sessions invalidated", ip=_client_ip())
    elif token:
        db.refresh_token_delete(token)
        db.audit_log(identity, "logout", "Single session", ip=_client_ip())
    else:
        db.audit_log(identity, "logout", "Access token discarded", ip=_client_ip())

    return jsonify({"message": "Logged out successfully."}), 200


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    """
    GET /api/auth/me
    Returns the current user's profile (no password hash).
    """
    identity = get_jwt_identity()
    user     = db.user_get_by_id(identity)
    if not user:
        return jsonify({"error": "User not found."}), 404

    user.pop("password_hash", None)
    user.pop("failed_attempts", None)
    user.pop("locked_until", None)
    user["id"] = str(user.pop("_id"))
    return jsonify(user), 200


@auth_bp.route("/change-password", methods=["POST"])
@jwt_required()
def change_password():
    """
    POST /api/auth/change-password
    Body (JSON): current_password, new_password
    """
    data             = request.get_json(silent=True) or {}
    current_password = data.get("current_password", "")
    new_password     = data.get("new_password", "")
    identity         = get_jwt_identity()

    user = db.user_get_by_id(identity)
    if not user:
        return jsonify({"error": "User not found."}), 404

    if not verify_password(current_password, user["password_hash"]):
        db.audit_log(identity, "change_password_fail",
                     "Wrong current password", ip=_client_ip(), success=False)
        return jsonify({"error": "Current password is incorrect."}), 401

    errors = validate_password_strength(new_password)
    if errors:
        return jsonify({"error": "Weak new password.", "details": errors}), 400

    new_hash = hash_password(new_password)
    db.users_col.update_one(
        {"_id": user["_id"]},
        {"$set": {"password_hash": new_hash}}
    )
    # Invalidate all sessions on password change
    db.refresh_tokens_delete_all_for_user(identity)
    db.audit_log(identity, "change_password", "Password updated; all sessions revoked",
                 ip=_client_ip())
    return jsonify({"message": "Password changed. Please sign in again."}), 200


# ─────────────────────────────────────────
#  Blueprint: /api/admin  (admin-only user management)
# ─────────────────────────────────────────

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


@admin_bp.route("/users", methods=["GET"])
@jwt_required()
@require_role("admin")
def list_users():
    """GET /api/admin/users?page=1&limit=20"""
    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 20))
    return jsonify(db.user_list(page, limit)), 200


@admin_bp.route("/users/<user_id>/deactivate", methods=["POST"])
@jwt_required()
@require_role("admin")
def deactivate_user(user_id):
    """POST /api/admin/users/<id>/deactivate"""
    db.users_col.update_one(
        {"_id": db.ObjectId(user_id)},
        {"$set": {"is_active": False}}
    )
    db.refresh_tokens_delete_all_for_user(user_id)
    db.audit_log(g.user_id, "deactivate_user", f"Deactivated: {user_id}", ip=_client_ip())
    return jsonify({"message": "User deactivated."}), 200


@admin_bp.route("/users/<user_id>/role", methods=["PATCH"])
@jwt_required()
@require_role("admin")
def update_role(user_id):
    """PATCH /api/admin/users/<id>/role  Body: {role: 'inspector'}"""
    data = request.get_json(silent=True) or {}
    role = data.get("role", "").lower()
    if role not in ROLE_HIERARCHY:
        return jsonify({"error": f"Invalid role. Choose from: {', '.join(ROLE_HIERARCHY)}"}), 400
    db.users_col.update_one(
        {"_id": db.ObjectId(user_id)},
        {"$set": {"role": role}}
    )
    db.audit_log(g.user_id, "update_role",
                 f"Set {user_id} → {role}", ip=_client_ip())
    return jsonify({"message": f"Role updated to '{role}'."}), 200


@admin_bp.route("/audit-log", methods=["GET"])
@jwt_required()
@require_role("admin")
def get_audit_log():
    """GET /api/admin/audit-log?page=1&limit=50"""
    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 50))
    skip  = (page - 1) * limit
    cursor = (
        db.audit_col
        .find({})
        .sort("timestamp", -1)
        .skip(skip)
        .limit(limit)
    )
    docs  = [{**d, "id": str(d.pop("_id"))} for d in cursor]
    total = db.audit_col.count_documents({})
    return jsonify({"logs": docs, "total": total, "page": page, "limit": limit}), 200


# ─────────────────────────────────────────
#  Shared utility
# ─────────────────────────────────────────

def _client_ip() -> str:
    """Best-effort real IP, respecting X-Forwarded-For behind a proxy."""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or ""
