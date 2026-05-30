"""
Run from pva2/ folder:  python patch_app.py
Inserts new routes into app.py before the if __name__ == "__main__" block.
"""

NEW_ROUTES = '''
# ── Reading mutations ────────────────────────────────────────────────────────
@app.route("/api/readings/<reading_id>", methods=["DELETE"])
def api_delete_reading(reading_id):
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import delete_reading
        if not delete_reading(payload["user_id"], reading_id):
            return jsonify({"message": "Reading not found."}), 404
        return jsonify({"deleted": True})
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/api/readings/<reading_id>", methods=["PATCH"])
def api_edit_reading(reading_id):
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    body = request.get_json(silent=True) or {}
    try:
        from src.database import update_reading_notes
        if not update_reading_notes(payload["user_id"], reading_id,
                                    body.get("notes", "")):
            return jsonify({"message": "Reading not found."}), 404
        return jsonify({"updated": True})
    except Exception as e:
        return jsonify({"message": str(e)}), 500


# ── Share links ───────────────────────────────────────────────────────────────
@app.route("/api/share", methods=["GET"])
def api_get_share():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import get_share_for_user
        token = get_share_for_user(payload["user_id"])
        if not token:
            return jsonify({"token": None, "url": None})
        base = request.host_url.rstrip("/")
        return jsonify({"token": token, "url": f"{base}/report/{token}"})
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/api/share", methods=["POST"])
def api_create_share():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import create_share_token
        token = create_share_token(payload["user_id"])
        base = request.host_url.rstrip("/")
        return jsonify({"token": token, "url": f"{base}/report/{token}"})
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/api/share", methods=["DELETE"])
def api_revoke_share():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import revoke_share
        revoke_share(payload["user_id"])
        return jsonify({"revoked": True})
    except Exception as e:
        return jsonify({"message": str(e)}), 500


# ── Shared report (public, no auth) ───────────────────────────────────────────
@app.route("/report/<token>")
def shared_report(token):
    return render_template("report.html", token=token)


@app.route("/api/report/<token>")
def api_shared_report(token):
    try:
        from src.database import get_share_by_token, get_readings, get_stats, get_user_by_id
        from datetime import datetime as _dt
        share = get_share_by_token(token)
        if not share:
            return jsonify({"message": "Report not found or link has expired."}), 404
        uid = share["user_id"]
        user = get_user_by_id(uid)
        readings = get_readings(uid)
        stats = get_stats(uid)
        return jsonify({
            "patient_name": user["name"] if user else "Anonymous",
            "readings": readings,
            "stats": stats,
            "generated_at": _dt.utcnow().isoformat(),
        })
    except Exception as e:
        return jsonify({"message": str(e)}), 500


# ── Print-to-PDF export ───────────────────────────────────────────────────────
@app.route("/dashboard/export")
def export_pdf():
    from src.auth import verify_token
    token = request.args.get("token", "")
    if not token:
        return "Unauthorized", 401
    payload = verify_token(token)
    if not payload:
        return "Invalid or expired token", 401
    try:
        from src.database import get_readings, get_stats, get_user_by_id
        from datetime import datetime as _dt
        user = get_user_by_id(payload["user_id"])
        readings = get_readings(payload["user_id"])
        stats = get_stats(payload["user_id"])
        return render_template("export.html", user=user, readings=readings,
                               stats=stats,
                               generated_at=_dt.utcnow().strftime("%d %b %Y %H:%M UTC"))
    except Exception as e:
        return str(e), 500

'''

path = "app.py"
content = open(path).read()

if "api_delete_reading" in content:
    print("Already patched — skipping")
else:
    marker = "\nif __name__ == \"__main__\":"
    idx = content.rfind(marker)
    if idx == -1:
        print("ERROR: could not find if __name__ marker")
    else:
        content = content[:idx] + NEW_ROUTES + content[idx:]
        open(path, "w").write(content)
        print(f"✓ Inserted 7 new routes into {path}")
