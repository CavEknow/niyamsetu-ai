"""
database.py — NiyamSetu AI
MongoDB connection, collections, indexes, and all DB helper functions.

Requires:
    pip install pymongo python-dotenv

.env variables needed:
    MONGO_URI=mongodb://localhost:27017/niyamsetu
"""

import os
from datetime import datetime
from typing import Optional

from bson import ObjectId
from pymongo import MongoClient, DESCENDING, ASCENDING, errors
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
#  Connection
# ─────────────────────────────────────────

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/niyamsetu")

try:
    import certifi
    _client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        tlsCAFile=certifi.where(),
        tls=True,
    )
    _client.admin.command("ping")
    print("[NiyamSetu] ✅  MongoDB connected successfully.")
except Exception as exc:
    print(f"[NiyamSetu] ❌  MongoDB connection failed: {exc}")
    print("[NiyamSetu] ⚠  App will start but DB features may not work.")
    _client = None

db = _client.get_default_database()

# ─────────────────────────────────────────
#  Collections
# ─────────────────────────────────────────

users_col        = db["users"]          # registered officers / admins
inspections_col  = db["inspections"]    # each label-scan job
violations_col   = db["violations"]     # individual rule failures
products_col     = db["products"]       # barcode → product master
refresh_col      = db["refresh_tokens"] # JWT refresh tokens
audit_col        = db["audit_log"]      # login / action audit trail

# ─────────────────────────────────────────
#  Indexes  (idempotent — safe to call on every startup)
# ─────────────────────────────────────────

def _create_indexes():
    # users
    users_col.create_index("email",    unique=True)
    users_col.create_index("badge_id", unique=True, sparse=True)

    # inspections
    inspections_col.create_index([("created_at", DESCENDING)])
    inspections_col.create_index("inspector_id")
    inspections_col.create_index("job_id", unique=True)
    inspections_col.create_index("summary.status")
    inspections_col.create_index("product_type")

    # violations
    violations_col.create_index("inspection_id")
    violations_col.create_index([("created_at", DESCENDING)])
    violations_col.create_index("rule_key")

    # products
    products_col.create_index("barcode", unique=True, sparse=True)

    # refresh tokens — auto-expire after 30 days
    refresh_col.create_index("expires_at", expireAfterSeconds=0)
    refresh_col.create_index("token", unique=True)

    # audit log
    audit_col.create_index([("timestamp", DESCENDING)])
    audit_col.create_index("user_id")

    print("[NiyamSetu] ✅  MongoDB indexes ensured.")

_create_indexes()

# ─────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────

def _to_str_id(doc: dict) -> dict:
    """Convert ObjectId '_id' to string 'id' for JSON serialisation."""
    if doc and "_id" in doc:
        doc["id"] = str(doc.pop("_id"))
    return doc


def utcnow() -> datetime:
    return datetime.utcnow()


# ─────────────────────────────────────────
#  USER helpers
# ─────────────────────────────────────────

def user_get_by_email(email: str) -> Optional[dict]:
    """Return the user document for *email*, or None."""
    return users_col.find_one({"email": email.lower().strip()})


def user_get_by_id(user_id: str) -> Optional[dict]:
    """Return a user document by its string ObjectId."""
    try:
        return users_col.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None


def user_create(name: str, email: str, password_hash: str,
                role: str = "inspector",
                department: str = "",
                badge_id: str = "") -> str:
    """
    Insert a new user. Returns the new user's string ObjectId.
    Raises pymongo.errors.DuplicateKeyError if email already exists.
    """
    doc = {
        "name":           name.strip(),
        "email":          email.lower().strip(),
        "password_hash":  password_hash,
        "role":           role,              # admin | inspector | viewer
        "department":     department.strip(),
        "badge_id":       badge_id.strip() or None,
        "is_active":      True,
        "created_at":     utcnow(),
        "last_login":     None,
        "login_count":    0,
        "failed_attempts":0,
        "locked_until":   None,
    }
    result = users_col.insert_one(doc)
    return str(result.inserted_id)


def user_update_login(user_id: str):
    """Stamp last_login and increment login_count after a successful login."""
    users_col.update_one(
        {"_id": ObjectId(user_id)},
        {
            "$set": {"last_login": utcnow(), "failed_attempts": 0, "locked_until": None},
            "$inc": {"login_count": 1},
        }
    )


def user_record_failed_attempt(user_id: str) -> int:
    """
    Increment failed_attempts. Returns the new count.
    Caller should lock the account if count >= 5.
    """
    result = users_col.find_one_and_update(
        {"_id": ObjectId(user_id)},
        {"$inc": {"failed_attempts": 1}},
        return_document=True,
    )
    return result.get("failed_attempts", 1) if result else 1


def user_lock(user_id: str, lock_until: datetime):
    """Lock a user account until *lock_until*."""
    users_col.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"locked_until": lock_until}},
    )


def user_list(page: int = 1, limit: int = 20) -> dict:
    """Paginated list of all users (admin use)."""
    skip = (page - 1) * limit
    cursor = users_col.find({}, {"password_hash": 0}).skip(skip).limit(limit)
    docs = [_to_str_id(d) for d in cursor]
    total = users_col.count_documents({})
    return {"users": docs, "total": total, "page": page, "limit": limit}


# ─────────────────────────────────────────
#  INSPECTION helpers
# ─────────────────────────────────────────

def inspection_save(data: dict, inspector_id: str) -> str:
    """
    Persist a completed inspection and extract individual violations
    into the violations collection.

    *data* is the dict returned by the rule engine (job_id, fields,
    results, summary, ocr_text, …).

    Returns the new inspection's string ObjectId.
    """
    doc = {
        **data,
        "inspector_id": inspector_id,
        "created_at":   utcnow(),
    }
    result = inspections_col.insert_one(doc)
    inspection_oid = str(result.inserted_id)

    # Persist individual FAIL items as violation documents
    fail_docs = []
    for r in data.get("results", []):
        if r.get("status") == "FAIL":
            fail_docs.append({
                "inspection_id": inspection_oid,
                "job_id":        data.get("job_id"),
                "rule_key":      r.get("key"),
                "rule_label":    r.get("label"),
                "severity":      _severity(r.get("key")),
                "detected_value": r.get("value"),
                "created_at":    utcnow(),
            })
    if fail_docs:
        violations_col.insert_many(fail_docs)

    return inspection_oid


def _severity(rule_key: str) -> str:
    """Map a rule key to a display severity level."""
    high = {"manufacturer", "mrp", "net_quantity", "consumer_care", "origin"}
    return "HIGH" if rule_key in high else "MEDIUM"


def inspection_get(job_id: str) -> Optional[dict]:
    doc = inspections_col.find_one({"job_id": job_id})
    return _to_str_id(doc) if doc else None


def inspection_get_by_oid(oid: str) -> Optional[dict]:
    try:
        doc = inspections_col.find_one({"_id": ObjectId(oid)})
        return _to_str_id(doc) if doc else None
    except Exception:
        return None


def inspection_list(
    inspector_id: Optional[str] = None,
    status: Optional[str] = None,
    product_type: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
) -> dict:
    """
    Paginated inspection history.
    Pass inspector_id=None to retrieve all (admin view).
    """
    query: dict = {}
    if inspector_id:
        query["inspector_id"] = inspector_id
    if status:
        query["summary.status"] = {"$regex": status, "$options": "i"}
    if product_type:
        query["product_type"] = product_type

    skip = (page - 1) * limit
    cursor = (
        inspections_col
        .find(query, {
            "job_id": 1, "created_at": 1, "summary": 1,
            "product_type": 1, "inspector_id": 1, "image_url": 1,
        })
        .sort("created_at", DESCENDING)
        .skip(skip)
        .limit(limit)
    )
    docs = [_to_str_id(d) for d in cursor]
    total = inspections_col.count_documents(query)
    return {"inspections": docs, "total": total, "page": page, "limit": limit}


# ─────────────────────────────────────────
#  DASHBOARD helpers
# ─────────────────────────────────────────

def dashboard_stats() -> dict:
    """
    Aggregate dashboard figures:
    - total inspections
    - pass / fail / review counts
    - top 5 violation types
    - last 7-day daily counts
    """
    total         = inspections_col.count_documents({})
    total_fail    = inspections_col.count_documents({"summary.fails": {"$gt": 0}})
    total_pass    = inspections_col.count_documents({"summary.fails": 0, "summary.reviews": 0})
    total_review  = inspections_col.count_documents({"summary.fails": 0, "summary.reviews": {"$gt": 0}})

    # Top violation rule keys
    pipeline_top = [
        {"$group": {"_id": "$rule_key", "count": {"$sum": 1}}},
        {"$sort":  {"count": -1}},
        {"$limit": 5},
    ]
    top_violations = list(violations_col.aggregate(pipeline_top))

    # Daily inspection counts for the past 7 days
    from datetime import timedelta
    pipeline_daily = [
        {
            "$match": {
                "created_at": {"$gte": utcnow() - timedelta(days=7)}
            }
        },
        {
            "$group": {
                "_id": {
                    "$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}
                },
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": ASCENDING}},
    ]
    daily_counts = list(inspections_col.aggregate(pipeline_daily))

    # Recent 5 inspections
    recent_cursor = (
        inspections_col
        .find({}, {"job_id": 1, "created_at": 1, "summary": 1, "product_type": 1})
        .sort("created_at", DESCENDING)
        .limit(5)
    )
    recent = [_to_str_id(d) for d in recent_cursor]

    return {
        "total":          total,
        "total_pass":     total_pass,
        "total_fail":     total_fail,
        "total_review":   total_review,
        "pass_rate":      round(total_pass / total * 100, 1) if total else 0,
        "top_violations": top_violations,
        "daily_counts":   daily_counts,
        "recent":         recent,
    }


# ─────────────────────────────────────────
#  PRODUCT (barcode) helpers
# ─────────────────────────────────────────

def product_upsert(barcode: str, name: str = "", manufacturer: str = "") -> str:
    """Insert or update a product master record. Returns the product's ObjectId."""
    result = products_col.find_one_and_update(
        {"barcode": barcode},
        {
            "$set": {
                "barcode":      barcode,
                "name":         name,
                "manufacturer": manufacturer,
                "last_checked": utcnow(),
            },
            "$setOnInsert": {"created_at": utcnow()},
            "$push": {
                "compliance_history": {
                    "$each": [{"timestamp": utcnow()}],
                    "$slice": -50,            # keep last 50 checks only
                }
            },
        },
        upsert=True,
        return_document=True,
    )
    return str(result["_id"])


def product_get(barcode: str) -> Optional[dict]:
    doc = products_col.find_one({"barcode": barcode})
    return _to_str_id(doc) if doc else None


# ─────────────────────────────────────────
#  REFRESH TOKEN helpers
# ─────────────────────────────────────────

def refresh_token_save(token: str, user_id: str, expires_at: datetime):
    refresh_col.insert_one({
        "token":      token,
        "user_id":    user_id,
        "expires_at": expires_at,
        "created_at": utcnow(),
    })


def refresh_token_get(token: str) -> Optional[dict]:
    return refresh_col.find_one({"token": token})


def refresh_token_delete(token: str):
    refresh_col.delete_one({"token": token})


def refresh_tokens_delete_all_for_user(user_id: str):
    """Invalidate ALL refresh tokens for a user (logout-everywhere)."""
    refresh_col.delete_many({"user_id": user_id})


# ─────────────────────────────────────────
#  AUDIT LOG helpers
# ─────────────────────────────────────────

def audit_log(user_id: Optional[str], action: str, detail: str = "",
              ip: str = "", success: bool = True):
    """Write an immutable audit entry."""
    audit_col.insert_one({
        "user_id":   user_id,
        "action":    action,        # e.g. "login", "logout", "inspect", "export"
        "detail":    detail,
        "ip":        ip,
        "success":   success,
        "timestamp": utcnow(),
    })
