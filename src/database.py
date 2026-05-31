"""
MongoDB operations for the Parkinson's Voice Analyser.
Handles user accounts and longitudinal reading storage.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, List, Dict

import pymongo
from bson import ObjectId

MONGODB_URI = os.environ.get("MONGODB_URI", "")
DB_NAME = "parkinsons_voice"

_client = None
_db = None


def get_db():
    global _client, _db
    if _db is not None:
        return _db
    if not MONGODB_URI:
        raise RuntimeError(
            "MONGODB_URI environment variable not set. "
            "Add it as a HuggingFace Space secret."
        )
    _client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
    _db = _client[DB_NAME]
    # Indexes — idempotent, safe to call every startup
    _db.users.create_index("email", unique=True)
    _db.readings.create_index("user_id")
    _db.readings.create_index([("user_id", 1), ("timestamp", -1)])
    return _db


def db_available() -> bool:
    try:
        get_db()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# User operations
# ---------------------------------------------------------------------------

def create_user(email: str, password_hash: str, name: str) -> Dict:
    db = get_db()
    doc = {
        "email": email.lower().strip(),
        "password_hash": password_hash,
        "name": name.strip(),
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow(),
    }
    result = db.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get_user_by_email(email: str) -> Optional[Dict]:
    db = get_db()
    return db.users.find_one({"email": email.lower().strip()})


def get_user_by_id(user_id: str) -> Optional[Dict]:
    db = get_db()
    try:
        return db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None


def update_last_login(user_id: str):
    db = get_db()
    db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"last_login": datetime.utcnow()}}
    )


# ---------------------------------------------------------------------------
# Reading operations
# ---------------------------------------------------------------------------

def save_reading(
    user_id: str,
    probability_pd: float,
    prediction: int,
    confidence_pct: float,
    audio_duration_s: float,
    backend: str,
    model: str,
    notes: str = "",
) -> Dict:
    db = get_db()
    doc = {
        "user_id": ObjectId(user_id),
        "timestamp": datetime.utcnow(),
        "probability_pd": round(probability_pd, 4),
        "prediction": prediction,
        "confidence_pct": round(confidence_pct, 1),
        "audio_duration_s": round(float(audio_duration_s or 0), 1),
        "backend": backend,
        "model": model,
        "notes": notes.strip()[:500],
    }
    result = db.readings.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    doc["user_id"] = str(doc["user_id"])
    doc["timestamp"] = doc["timestamp"].isoformat()
    return doc


def get_readings(user_id: str, limit: int = 200) -> List[Dict]:
    db = get_db()
    cursor = db.readings.find(
        {"user_id": ObjectId(user_id)},
        sort=[("timestamp", -1)],
        limit=limit,
    )
    readings = []
    for r in cursor:
        r["_id"] = str(r["_id"])
        r["user_id"] = str(r["user_id"])
        r["timestamp"] = r["timestamp"].isoformat()
        readings.append(r)
    return readings


def get_stats(user_id: str) -> Dict:
    readings = get_readings(user_id)
    if not readings:
        return {
            "total": 0,
            "avg_probability": None,
            "rolling_avg_7": None,
            "trend": "insufficient_data",
            "latest_probability": None,
        }

    # Readings arrive newest-first; reverse for chronological order
    probs_chrono = list(reversed([r["probability_pd"] for r in readings]))
    total = len(probs_chrono)
    avg = sum(probs_chrono) / total

    # Rolling average of last 7 readings
    last_7 = probs_chrono[-7:]
    rolling_avg = sum(last_7) / len(last_7)

    # Trend: compare first-third vs last-third
    trend = "insufficient_data"
    if total >= 6:
        third = max(total // 3, 2)
        first_avg = sum(probs_chrono[:third]) / third
        last_avg = sum(probs_chrono[-third:]) / third
        diff = last_avg - first_avg
        if diff > 0.05:
            trend = "worsening"
        elif diff < -0.05:
            trend = "improving"
        else:
            trend = "stable"
    elif total >= 2:
        if probs_chrono[-1] > probs_chrono[0] + 0.05:
            trend = "worsening"
        elif probs_chrono[-1] < probs_chrono[0] - 0.05:
            trend = "improving"
        else:
            trend = "stable"

    return {
        "total": total,
        "avg_probability": round(avg, 4),
        "rolling_avg_7": round(rolling_avg, 4),
        "trend": trend,
        "latest_probability": probs_chrono[-1],
    }


# ---------------------------------------------------------------------------
# Reading mutations
# ---------------------------------------------------------------------------

def delete_reading(user_id: str, reading_id: str) -> bool:
    """Delete a reading owned by user_id. Returns True if deleted."""
    db = get_db()
    try:
        result = db.readings.delete_one({
            "_id": ObjectId(reading_id),
            "user_id": ObjectId(user_id),
        })
        return result.deleted_count > 0
    except Exception:
        return False


def update_reading_notes(user_id: str, reading_id: str, notes: str) -> bool:
    """Update notes on a reading. Returns True if updated."""
    db = get_db()
    try:
        result = db.readings.update_one(
            {"_id": ObjectId(reading_id), "user_id": ObjectId(user_id)},
            {"$set": {"notes": notes.strip()[:500]}},
        )
        return result.modified_count > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Shareable report links
# ---------------------------------------------------------------------------

def create_share_token(user_id: str) -> str:
    """Create or refresh a 30-day share token for a user."""
    import secrets as _sec
    from datetime import timedelta
    db = get_db()
    token = _sec.token_urlsafe(24)
    expires = datetime.utcnow() + timedelta(days=30)
    db.shares.update_one(
        {"user_id": ObjectId(user_id)},
        {"$set": {"token": token, "expires_at": expires,
                  "created_at": datetime.utcnow()}},
        upsert=True,
    )
    # Index for fast lookup by token
    try:
        db.shares.create_index("token", unique=True)
    except Exception:
        pass
    return token


def get_share_by_token(token: str) -> Optional[Dict]:
    """Return share doc if token is valid and not expired."""
    db = get_db()
    share = db.shares.find_one({"token": token})
    if not share:
        return None
    if share["expires_at"] < datetime.utcnow():
        return None
    share["user_id"] = str(share["user_id"])
    return share


def get_share_for_user(user_id: str) -> Optional[str]:
    """Return active share token for user, or None."""
    db = get_db()
    share = db.shares.find_one({"user_id": ObjectId(user_id)})
    if not share or share["expires_at"] < datetime.utcnow():
        return None
    return share["token"]


def revoke_share(user_id: str):
    """Delete the user's share token."""
    get_db().shares.delete_one({"user_id": ObjectId(user_id)})


# ---------------------------------------------------------------------------
# Motor readings
# ---------------------------------------------------------------------------

def save_motor_reading(
    user_id: str,
    spiral_score: float,
    spiral_pd_risk: float,
    spiral_features: dict,
    typing_score: float,
    typing_pd_risk: float,
    typing_features: dict,
    combined_probability: float,
    combined_prediction: int,
    voice_probability,
    notes: str = "",
) -> Dict:
    db = get_db()
    try:
        db.motor_readings.create_index("user_id")
        db.motor_readings.create_index([("user_id", 1), ("timestamp", -1)])
    except Exception:
        pass
    doc = {
        "user_id":             ObjectId(user_id),
        "timestamp":           datetime.utcnow(),
        "spiral_score":        round(float(spiral_score), 2),
        "spiral_pd_risk":      round(float(spiral_pd_risk), 4),
        "spiral_features":     spiral_features,
        "typing_score":        round(float(typing_score), 2),
        "typing_pd_risk":      round(float(typing_pd_risk), 4),
        "typing_features":     typing_features,
        "combined_probability": round(float(combined_probability), 4),
        "combined_prediction": int(combined_prediction),
        "voice_probability":   round(float(voice_probability), 4) if voice_probability is not None else None,
        "notes":               notes.strip()[:500],
    }
    result = db.motor_readings.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    doc["user_id"] = str(doc["user_id"])
    doc["timestamp"] = doc["timestamp"].isoformat()
    return doc


def get_motor_readings(user_id: str, limit: int = 100) -> List[Dict]:
    db = get_db()
    cursor = db.motor_readings.find(
        {"user_id": ObjectId(user_id)},
        sort=[("timestamp", -1)],
        limit=limit,
    )
    readings = []
    for r in cursor:
        r["_id"] = str(r["_id"])
        r["user_id"] = str(r["user_id"])
        r["timestamp"] = r["timestamp"].isoformat()
        readings.append(r)
    return readings


def get_motor_stats(user_id: str) -> Dict:
    readings = get_motor_readings(user_id)
    if not readings:
        return {
            "total": 0,
            "avg_combined": None,
            "avg_spiral_score": None,
            "avg_typing_score": None,
            "trend": "insufficient_data",
            "latest_combined": None,
        }
    combined = list(reversed([r["combined_probability"] for r in readings]))
    total = len(combined)
    avg_combined = round(float(sum(combined) / total), 4)
    avg_spiral = round(float(sum(r["spiral_score"] for r in readings) / total), 2)
    avg_typing = round(float(sum(r["typing_score"] for r in readings) / total), 2)

    trend = "insufficient_data"
    if total >= 4:
        half = total // 2
        first_avg = sum(combined[:half]) / half
        last_avg  = sum(combined[half:]) / (total - half)
        diff = last_avg - first_avg
        trend = "worsening" if diff > 0.05 else "improving" if diff < -0.05 else "stable"
    elif total >= 2:
        diff = combined[-1] - combined[0]
        trend = "worsening" if diff > 0.05 else "improving" if diff < -0.05 else "stable"

    return {
        "total":            total,
        "avg_combined":     avg_combined,
        "avg_spiral_score": avg_spiral,
        "avg_typing_score": avg_typing,
        "trend":            trend,
        "latest_combined":  combined[-1] if combined else None,
    }
