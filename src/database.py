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
