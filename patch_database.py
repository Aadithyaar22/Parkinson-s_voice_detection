"""
Run from pva2/ folder:  python patch_database.py
Appends new functions to src/database.py
"""
ADDITIONS = '''

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
    """Delete the user\'s share token."""
    get_db().shares.delete_one({"user_id": ObjectId(user_id)})
'''

path = "src/database.py"
content = open(path).read()
if "def delete_reading" in content:
    print("Already patched — skipping")
else:
    with open(path, "a") as f:
        f.write(ADDITIONS)
    print(f"✓ Appended 6 new functions to {path}")
