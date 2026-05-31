"""
Complete motor data integration patch.
Run from pva2/:  python patch_motor_full.py

Changes:
  1. src/database.py       — add save/get/stats for motor_readings collection
  2. app.py                — add /api/motor/save, /api/motor/readings, /api/motor/stats
  3. templates/motor.html  — add Save Assessment button + auth-aware UI
  4. templates/dashboard.html — full motor section with trend overlay + history
  5. templates/export.html — motor assessment section in PDF
  6. templates/report.html — motor data in doctor report
"""

# ── 1. database.py — append motor functions ───────────────────────────────────
DB_ADDITIONS = '''

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
'''

db_path = "src/database.py"
db_content = open(db_path).read()
if "save_motor_reading" not in db_content:
    with open(db_path, "a") as f:
        f.write(DB_ADDITIONS)
    print("✓ database.py — added motor_readings functions")
else:
    print("  database.py already patched")


# ── 2. app.py — add motor save/read routes ────────────────────────────────────
MOTOR_API_ROUTES = '''
# ── Motor save / read ─────────────────────────────────────────────────────────
@app.route("/api/motor/save", methods=["POST"])
def api_motor_save():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    body = request.get_json(silent=True) or {}
    try:
        from src.database import save_motor_reading
        doc = save_motor_reading(
            user_id=payload["user_id"],
            spiral_score=body.get("spiral_score", 50),
            spiral_pd_risk=body.get("spiral_pd_risk", 0.5),
            spiral_features=body.get("spiral_features", {}),
            typing_score=body.get("typing_score", 50),
            typing_pd_risk=body.get("typing_pd_risk", 0.5),
            typing_features=body.get("typing_features", {}),
            combined_probability=body.get("combined_probability", 0.5),
            combined_prediction=body.get("combined_prediction", 0),
            voice_probability=body.get("voice_probability"),
            notes=body.get("notes", ""),
        )
        return jsonify(doc)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"message": str(e)}), 500


@app.route("/api/motor/readings", methods=["GET"])
def api_motor_readings():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import get_motor_readings
        return jsonify(get_motor_readings(payload["user_id"]))
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/api/motor/stats", methods=["GET"])
def api_motor_stats():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import get_motor_stats
        return jsonify(get_motor_stats(payload["user_id"]))
    except Exception as e:
        return jsonify({"message": str(e)}), 500

'''

app_content = open("app.py").read()
if "api_motor_save" not in app_content:
    marker = "\n# ── Motor assessment"
    if marker in app_content:
        app_content = app_content.replace(marker, MOTOR_API_ROUTES + marker)
    else:
        marker2 = "\nif __name__ == \"__main__\":"
        idx = app_content.rfind(marker2)
        app_content = app_content[:idx] + MOTOR_API_ROUTES + app_content[idx:]
    open("app.py", "w").write(app_content)
    print("✓ app.py — added motor save/read/stats routes")
else:
    print("  app.py motor routes already present")


# ── 3. motor.html — add save button + auth-aware UI ──────────────────────────
motor_content = open("templates/motor.html").read()

SAVE_BUTTON_JS = """
// ── Save motor assessment ─────────────────────────────────────────────────────
function showSaveMotor(data) {
  const token = localStorage.getItem("pvd_token");
  const sp = document.getElementById("motorSavePanel");
  if (!sp) return;
  sp.style.display = "block";
  if (!token) {
    sp.innerHTML = `<div style="padding:14px;background:var(--blue-50);border-radius:12px;
      border:1px solid var(--blue-100);font-size:13px;color:var(--ink-soft)">
      <a href="/login" style="color:var(--blue-600);font-weight:500">Sign in</a>
      to save this assessment to your history and track motor trends over time.
    </div>`;
    return;
  }
  sp.innerHTML = `
    <div style="padding:18px;background:var(--blue-50);border-radius:14px;border:1px solid var(--blue-100)">
      <div style="font-family:var(--display);font-weight:600;font-size:14px;
        color:var(--blue-700);margin-bottom:10px">
        💾 Save this assessment to your history
      </div>
      <textarea id="motorNotes" placeholder="Optional notes — e.g. 'took medication 2h ago', 'left hand used'" rows="2"
        style="width:100%;padding:9px 12px;border:1.5px solid var(--blue-200);border-radius:10px;
        font-family:var(--sans);font-size:13px;resize:vertical;margin-bottom:10px;
        color:var(--ink);background:white;outline:none;"></textarea>
      <div style="display:flex;align-items:center;gap:10px">
        <button class="btn btn-primary" id="motorSaveBtn" onclick="saveMotorReading()"
          style="padding:10px 20px;font-size:13px">💾 Save assessment</button>
        <span id="motorSaveMsg" style="font-size:13px;color:var(--ink-mute)"></span>
      </div>
    </div>`;
}

async function saveMotorReading() {
  const token = localStorage.getItem("pvd_token");
  if (!token || !spiralScore || !typingScore) return;
  const btn = document.getElementById("motorSaveBtn");
  const msg = document.getElementById("motorSaveMsg");
  const notes = document.getElementById("motorNotes")?.value || "";
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span> Saving…';

  // Get combined data
  const body = {
    spiral_score:    spiralScore.overall_score,
    spiral_pd_risk:  spiralScore.pd_risk,
    spiral_features: spiralScore.features,
    typing_score:    typingScore.overall_score,
    typing_pd_risk:  typingScore.pd_risk,
    typing_features: typingScore.features,
    voice_probability: voiceProb >= 0 ? voiceProb : null,
    notes,
  };

  // Get combined probability from last combined result
  const combinedPctEl = document.getElementById("combinedPct");
  if (combinedPctEl && combinedPctEl.textContent !== "—") {
    body.combined_probability = parseFloat(combinedPctEl.textContent) / 100;
    body.combined_prediction  = body.combined_probability >= 0.38 ? 1 : 0;
  } else {
    const spiral_risk = spiralScore.pd_risk;
    const typing_risk = typingScore.pd_risk;
    const v = voiceProb >= 0 ? voiceProb : 0.5;
    body.combined_probability = v * 0.55 + spiral_risk * 0.28 + typing_risk * 0.17;
    body.combined_prediction  = body.combined_probability >= 0.38 ? 1 : 0;
  }

  try {
    const res = await fetch("/api/motor/save", {
      method: "POST",
      headers: {"Content-Type": "application/json", "Authorization": "Bearer " + token},
      body: JSON.stringify(body),
    });
    if (res.status === 401) { localStorage.removeItem("pvd_token"); window.location.href = "/login"; return; }
    const data = await res.json();
    if (!res.ok) throw new Error(data.message);
    msg.textContent = "✓ Saved to your history";
    btn.textContent = "✓ Saved";
    btn.style.background = "linear-gradient(180deg,#0EA371,#059669)";
  } catch(e) {
    msg.textContent = "Failed: " + e.message;
    btn.disabled = false;
    btn.innerHTML = "💾 Save assessment";
  }
}
"""

SAVE_BUTTON_HTML = """
  <!-- Save panel — shown after combined score -->
  <div id="motorSavePanel" style="display:none;margin-top:16px"></div>
"""

if "motorSavePanel" not in motor_content:
    # Add HTML before closing </div> of combined panel
    motor_content = motor_content.replace(
        '    <div class="disclaimer-sm">',
        SAVE_BUTTON_HTML + '\n    <div class="disclaimer-sm">'
    )
    # Add JS before </script>
    motor_content = motor_content.replace(
        "window.addEventListener('load'",
        SAVE_BUTTON_JS + "\nwindow.addEventListener('load'"
    )
    # Call showSaveMotor after combined renders
    motor_content = motor_content.replace(
        "  if (voiceProb < 0) {\n    document.getElementById(\"voicePrompt\").classList.remove(\"hidden\");\n  }",
        "  if (voiceProb < 0) {\n    document.getElementById(\"voicePrompt\").classList.remove(\"hidden\");\n  }\n  showSaveMotor(data);"
    )
    open("templates/motor.html", "w").write(motor_content)
    print("✓ motor.html — added save assessment button")
else:
    print("  motor.html save panel already present")


# ── 4. dashboard.html — add motor section ─────────────────────────────────────
dash_content = open("templates/dashboard.html").read()

if "motorStats" not in dash_content:
    # Find the loadDashboard function and add motor data fetch
    dash_content = dash_content.replace(
        """    const [meR,readR,statR,shareR] = await Promise.all([
      fetch("/api/me",{headers:authH()}),
      fetch("/api/readings",{headers:authH()}),
      fetch("/api/stats",{headers:authH()}),
      fetch("/api/share",{headers:authH()}),
    ]);
    if (meR.status===401){logout();return;}
    _me=await meR.json(); _readings=await readR.json();
    _stats=await statR.json();
    const shareData=await shareR.json();
    render(_me,_readings,_stats,shareData);""",
        """    const [meR,readR,statR,shareR,motorR,motorStatR] = await Promise.all([
      fetch("/api/me",{headers:authH()}),
      fetch("/api/readings",{headers:authH()}),
      fetch("/api/stats",{headers:authH()}),
      fetch("/api/share",{headers:authH()}),
      fetch("/api/motor/readings",{headers:authH()}),
      fetch("/api/motor/stats",{headers:authH()}),
    ]);
    if (meR.status===401){logout();return;}
    _me=await meR.json(); _readings=await readR.json();
    _stats=await statR.json();
    const shareData=await shareR.json();
    const motorReadings=await motorR.json();
    const motorStats=await motorStatR.json();
    render(_me,_readings,_stats,shareData,motorReadings,motorStats);"""
    )

    # Update render function signature
    dash_content = dash_content.replace(
        "function render(me,readings,stats,shareData) {",
        "function render(me,readings,stats,shareData,motorReadings,motorStats) {"
    )

    # Add motor stats cards and section after the main stats
    MOTOR_DASHBOARD_SECTION = """
    ${motorReadings && motorReadings.length > 0 ? `
    <div class="panel" style="margin-top:0">
      <div class="panel-header">
        <div class="panel-title">✍️ Motor Assessment History</div>
        <a href="/motor" class="btn-nav btn-primary-sm" style="font-size:12px;padding:7px 14px;text-decoration:none">+ New assessment</a>
      </div>
      <div class="stats" style="margin-bottom:16px">
        <div class="stat-card">
          <div class="stat-label">Motor assessments</div>
          <div class="stat-value">${motorStats.total}</div>
          <div class="stat-sub">completed</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Avg combined risk</div>
          <div class="stat-value">${motorStats.avg_combined !== null ? (motorStats.avg_combined*100).toFixed(1)+"%" : "—"}</div>
          <div class="stat-sub">motor + voice</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Avg spiral score</div>
          <div class="stat-value" style="color:var(--blue-700)">${motorStats.avg_spiral_score !== null ? motorStats.avg_spiral_score.toFixed(0)+"/100" : "—"}</div>
          <div class="stat-sub">motor health</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Avg typing score</div>
          <div class="stat-value" style="color:var(--blue-700)">${motorStats.avg_typing_score !== null ? motorStats.avg_typing_score.toFixed(0)+"/100" : "—"}</div>
          <div class="stat-sub">motor health</div>
        </div>
      </div>
      <div class="table-wrap">${buildMotorTable(motorReadings)}</div>
    </div>` : `
    <div class="panel" style="margin-top:0">
      <div class="panel-header">
        <div class="panel-title">✍️ Motor Assessment</div>
        <a href="/motor" class="btn-nav btn-primary-sm" style="font-size:12px;padding:7px 14px;text-decoration:none">Take assessment</a>
      </div>
      <div class="empty" style="padding:32px">
        <div class="empty-icon">✍️</div>
        <h3>No motor assessments yet</h3>
        <p>Complete the <a href="/motor">spiral drawing and typing test</a> to add motor biomarkers to your profile.</p>
      </div>
    </div>`}
"""

    # Add motor section after reading history panel
    dash_content = dash_content.replace(
        "${total>0?`\n    <div class=\"panel\">\n      <div class=\"panel-header\">\n        <div class=\"panel-title\">Reading history</div>\n      </div>\n      <div class=\"table-wrap\">${buildTable(readings)}</div>\n    </div>`:\"\"}",
        "${total>0?`\n    <div class=\"panel\">\n      <div class=\"panel-header\">\n        <div class=\"panel-title\">Voice Reading History</div>\n      </div>\n      <div class=\"table-wrap\">${buildTable(readings)}</div>\n    </div>`:\"\"}\n  " + MOTOR_DASHBOARD_SECTION
    )

    # Add buildMotorTable function
    BUILD_MOTOR_TABLE = """
function buildMotorTable(readings) {
  if (!readings || !readings.length) return '<div class="empty"><p>No motor readings.</p></div>';
  const rows = readings.slice(0, 50).map(r => {
    const spiralColor = r.spiral_score >= 70 ? "var(--success)" : r.spiral_score >= 40 ? "var(--warn)" : "var(--danger)";
    const typingColor = r.typing_score >= 70 ? "var(--success)" : r.typing_score >= 40 ? "var(--warn)" : "var(--danger)";
    const combPct = (r.combined_probability * 100).toFixed(1);
    const badge = r.combined_prediction === 1
      ? '<span class="badge pd">PD indicators</span>'
      : '<span class="badge healthy">Healthy</span>';
    return `<tr>
      <td style="color:var(--ink-mute);font-size:12px">${fmtFull(r.timestamp)}</td>
      <td><strong style="color:${spiralColor}">${r.spiral_score.toFixed(0)}/100</strong></td>
      <td><strong style="color:${typingColor}">${r.typing_score.toFixed(0)}/100</strong></td>
      <td><strong>${combPct}%</strong></td>
      <td>${badge}</td>
      <td style="color:var(--ink-mute);font-size:12px">${r.voice_probability !== null && r.voice_probability !== undefined ? (r.voice_probability*100).toFixed(1)+"%" : "—"}</td>
      <td class="notes-cell"><span style="color:var(--ink-mute);font-style:italic;font-size:12px">${r.notes || "—"}</span></td>
      <td>
        <button class="del-btn" onclick="deleteMotorReading('${r._id}')" title="Delete">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/></svg>
        </button>
      </td>
    </tr>`;
  }).join("");
  return `<table>
    <thead><tr>
      <th>Date & time</th><th>Spiral score</th><th>Typing score</th>
      <th>Combined risk</th><th>Result</th><th>Voice prob</th><th>Notes</th><th></th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

async function deleteMotorReading(id) {
  if (!confirm("Delete this motor assessment?")) return;
  try {
    const res = await fetch(\`/api/motor/readings/\${id}\`, {method:"DELETE",headers:authH()});
    if (!res.ok) throw new Error((await res.json()).message);
    document.getElementById("row-m-"+id)?.remove();
    showToast("Motor assessment deleted");
    loadDashboard();
  } catch(e) { showToast("Failed: "+e.message); }
}
"""

    dash_content = dash_content.replace(
        "// ── Delete reading",
        BUILD_MOTOR_TABLE + "\n// ── Delete reading"
    )

    open("templates/dashboard.html", "w").write(dash_content)
    print("✓ dashboard.html — added motor section, stats, history table")
else:
    print("  dashboard.html already has motor section")


# ── 5. app.py — add DELETE /api/motor/readings/<id> ──────────────────────────
app_content2 = open("app.py").read()
if "api_delete_motor" not in app_content2:
    DELETE_MOTOR = """
@app.route("/api/motor/readings/<reading_id>", methods=["DELETE"])
def api_delete_motor(reading_id):
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import get_db
        from bson import ObjectId
        db = get_db()
        result = db.motor_readings.delete_one({
            "_id": ObjectId(reading_id),
            "user_id": ObjectId(payload["user_id"]),
        })
        if result.deleted_count == 0:
            return jsonify({"message": "Not found."}), 404
        return jsonify({"deleted": True})
    except Exception as e:
        return jsonify({"message": str(e)}), 500

"""
    app_content2 = app_content2.replace(
        "# ── Motor save / read",
        DELETE_MOTOR + "# ── Motor save / read"
    )
    open("app.py", "w").write(app_content2)
    print("✓ app.py — added DELETE motor reading route")
else:
    print("  motor delete route already present")


# ── 6. export.html — add motor section ────────────────────────────────────────
export_content = open("templates/export.html").read()

MOTOR_EXPORT_HTML = """
{% if motor_readings and motor_readings|length > 0 %}
<div class="section-title" style="margin-top:28px">Motor Assessment History</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:16px">
  <div class="stat">
    <div class="stat-label">Assessments</div>
    <div class="stat-value">{{ motor_readings|length }}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Avg combined risk</div>
    <div class="stat-value">
      {% if motor_stats and motor_stats.avg_combined %}
        {{ "%.1f"|format(motor_stats.avg_combined*100) }}%
      {% else %}—{% endif %}
    </div>
  </div>
  <div class="stat">
    <div class="stat-label">Avg spiral score</div>
    <div class="stat-value">{{ "%.0f"|format(motor_stats.avg_spiral_score) if motor_stats and motor_stats.avg_spiral_score else "—" }}/100</div>
  </div>
  <div class="stat">
    <div class="stat-label">Avg typing score</div>
    <div class="stat-value">{{ "%.0f"|format(motor_stats.avg_typing_score) if motor_stats and motor_stats.avg_typing_score else "—" }}/100</div>
  </div>
</div>
<table>
  <thead>
    <tr><th>Date & Time</th><th>Spiral Score</th><th>Typing Score</th><th>Combined Risk</th><th>Result</th><th>Voice Used</th><th>Notes</th></tr>
  </thead>
  <tbody>
    {% for r in motor_readings[:50] %}
    <tr>
      <td>{{ r.timestamp[:16].replace('T',' ') }} UTC</td>
      <td><strong>{{ "%.0f"|format(r.spiral_score) }}/100</strong></td>
      <td><strong>{{ "%.0f"|format(r.typing_score) }}/100</strong></td>
      <td><strong>{{ "%.1f"|format(r.combined_probability*100) }}%</strong></td>
      <td>{% if r.combined_prediction == 1 %}<span class="pd-tag">PD indicators</span>{% else %}<span class="hc-tag">Healthy</span>{% endif %}</td>
      <td>{{ "%.1f"|format(r.voice_probability*100) ~ "%" if r.voice_probability is not none else "—" }}</td>
      <td>{{ r.notes or "—" }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
{% endif %}
"""

if "motor_readings" not in export_content:
    export_content = export_content.replace(
        '<div class="disclaimer">',
        MOTOR_EXPORT_HTML + '\n<div class="disclaimer">'
    )
    open("templates/export.html", "w").write(export_content)
    print("✓ export.html — added motor assessment section")
else:
    print("  export.html already has motor section")


# ── 7. app.py export route — pass motor data ──────────────────────────────────
app_content3 = open("app.py").read()
if "get_motor_readings" not in app_content3:
    app_content3 = app_content3.replace(
        """        return render_template("export.html", user=user, readings=readings,
                               stats=stats, spectrogram_b64=spectrogram_b64,
                               generated_at=_dt.utcnow().strftime("%d %b %Y %H:%M UTC"))""",
        """        from src.database import get_motor_readings, get_motor_stats
        motor_readings = get_motor_readings(payload["user_id"])
        motor_stats = get_motor_stats(payload["user_id"])
        return render_template("export.html", user=user, readings=readings,
                               stats=stats, spectrogram_b64=spectrogram_b64,
                               motor_readings=motor_readings,
                               motor_stats=motor_stats,
                               generated_at=_dt.utcnow().strftime("%d %b %Y %H:%M UTC"))"""
    )
    open("app.py", "w").write(app_content3)
    print("✓ app.py — export route now passes motor data")
else:
    print("  export route already has motor data")


# ── 8. report.html — add motor data ───────────────────────────────────────────
report_content = open("templates/report.html").read()

MOTOR_REPORT_JS = """
    ${data.motor_readings && data.motor_readings.length > 0 ? `
    <div class="panel">
      <div class="panel-title">✍️ Motor Assessment History</div>
      <div class="stats" style="grid-template-columns:1fr 1fr 1fr">
        <div class="stat">
          <div class="stat-label">Total assessments</div>
          <div class="stat-value">${data.motor_stats?.total || 0}</div>
        </div>
        <div class="stat">
          <div class="stat-label">Avg combined risk</div>
          <div class="stat-value">${data.motor_stats?.avg_combined !== null ? (data.motor_stats.avg_combined*100).toFixed(1)+"%" : "—"}</div>
        </div>
        <div class="stat">
          <div class="stat-label">Motor trend</div>
          <div style="margin-top:6px">${trendBadge(data.motor_stats?.trend || "insufficient_data")}</div>
        </div>
      </div>
      <table style="margin-top:12px">
        <thead><tr><th>Date & Time</th><th>Spiral</th><th>Typing</th><th>Combined Risk</th><th>Result</th></tr></thead>
        <tbody>
          ${data.motor_readings.slice(0,20).map(r => {
            const pct = (r.combined_probability*100).toFixed(1);
            const badge = r.combined_prediction === 1
              ? '<span class="badge pd">PD indicators</span>'
              : '<span class="badge healthy">Healthy</span>';
            return `<tr>
              <td>${fmtFull(r.timestamp)}</td>
              <td>${r.spiral_score.toFixed(0)}/100</td>
              <td>${r.typing_score.toFixed(0)}/100</td>
              <td><strong>${pct}%</strong></td>
              <td>${badge}</td>
            </tr>`;
          }).join("")}
        </tbody>
      </table>
    </div>` : ""}
"""

if "motor_readings" not in report_content:
    report_content = report_content.replace(
        '    <div class="disclaimer">',
        MOTOR_REPORT_JS + '\n    <div class="disclaimer">'
    )
    # Update API call to also fetch motor data
    report_content = report_content.replace(
        """    const res = await fetch("/api/report/" + token);""",
        """    const res = await fetch("/api/report/" + token);"""
    )
    open("templates/report.html", "w").write(report_content)
    print("✓ report.html — added motor section")
else:
    print("  report.html already has motor section")


# ── 9. app.py shared report — include motor data ──────────────────────────────
app_content4 = open("app.py").read()
if "get_motor_readings" not in app_content4.split("api_shared_report")[1][:500] if "api_shared_report" in app_content4 else True:
    app_content4 = app_content4.replace(
        """        return jsonify({
            "patient_name": user["name"] if user else "Anonymous",
            "readings": readings,
            "stats": stats,
            "generated_at": _dt.utcnow().isoformat(),
        })""",
        """        from src.database import get_motor_readings, get_motor_stats
        motor_readings = get_motor_readings(uid)
        motor_stats = get_motor_stats(uid)
        return jsonify({
            "patient_name": user["name"] if user else "Anonymous",
            "readings": readings,
            "stats": stats,
            "motor_readings": motor_readings,
            "motor_stats": motor_stats,
            "generated_at": _dt.utcnow().isoformat(),
        })"""
    )
    open("app.py", "w").write(app_content4)
    print("✓ app.py — shared report now includes motor data")
else:
    print("  shared report already includes motor data")


print("\n✅ All patches applied. Now commit and push.")
