"""
Run from pva2/:  python patch_motor.py
Adds 4 things:
  1. /motor page route in app.py
  2. /api/motor/spiral, /api/motor/typing, /api/motor/combined routes in app.py
  3. Motor Assessment link in index.html nav
  4. JS to save voice probability to localStorage for motor page to use
"""

# ── 1. Patch app.py ───────────────────────────────────────────────────────────
MOTOR_ROUTES = '''
# ── Motor assessment ──────────────────────────────────────────────────────────
@app.route("/motor")
def motor_page():
    return render_template("motor.html")


@app.route("/api/motor/spiral", methods=["POST"])
def api_motor_spiral():
    body = request.get_json(silent=True) or {}
    try:
        from src.motor_analyzer import score_spiral
        result = score_spiral({
            "velocity_cv":   body.get("velocity_cv"),
            "tremor_freq":   body.get("tremor_freq"),
            "deviation_norm": body.get("deviation_norm"),
        })
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/motor/typing", methods=["POST"])
def api_motor_typing():
    body = request.get_json(silent=True) or {}
    try:
        from src.motor_analyzer import score_typing
        result = score_typing({
            "iki_cv":     body.get("iki_cv"),
            "wpm":        body.get("wpm"),
            "hold_cv":    body.get("hold_cv"),
            "error_rate": body.get("error_rate"),
        })
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/motor/combined", methods=["POST"])
def api_motor_combined():
    body = request.get_json(silent=True) or {}
    try:
        from src.motor_analyzer import compute_combined_score
        voice_prob   = body.get("voice_probability")
        spiral_score = float(body.get("spiral_score", 50))
        typing_score = float(body.get("typing_score", 50))
        # If no voice reading, use neutral 0.5
        if voice_prob is None or float(voice_prob) < 0:
            voice_prob = 0.5
        result = compute_combined_score(
            float(voice_prob), spiral_score, typing_score
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

'''

app_content = open("app.py").read()
if "api_motor_spiral" in app_content:
    print("app.py already patched — skipping motor routes")
else:
    marker = "\nif __name__ == \"__main__\":"
    idx = app_content.rfind(marker)
    app_content = app_content[:idx] + MOTOR_ROUTES + app_content[idx:]
    open("app.py", "w").write(app_content)
    print("✓ Added 4 motor routes to app.py")


# ── 2. Patch index.html ───────────────────────────────────────────────────────
idx_content = open("templates/index.html").read()

# Add Motor Assessment link in nav pills
motor_pill = """        <a href="/motor" class="pill" style="text-decoration:none;background:linear-gradient(135deg,#F5F3FF,#EFF6FF);border-color:#DDD6FE;color:#5B21B6">✍️ Motor test</a>"""

if "Motor test" not in idx_content:
    idx_content = idx_content.replace(
        "      {% endif %}\n    </div>\n  </header>",
        "      {% endif %}\n" + motor_pill + "\n    </div>\n  </header>"
    )
    print("✓ Added Motor Assessment link to nav")
else:
    print("Motor link already in nav — skipping")

# Add JS to save voice probability to localStorage after each analysis
save_voice_prob_js = """
  // Save voice probability for motor assessment page
  localStorage.setItem("pvd_last_voice_prob", r.probability_pd.toString());
"""

if "pvd_last_voice_prob" not in idx_content:
    idx_content = idx_content.replace(
        "  showSavePanel();\n  updateAuthUI();",
        "  showSavePanel();\n  updateAuthUI();\n" + save_voice_prob_js
    )
    print("✓ Added voice prob save to localStorage")
else:
    print("Voice prob save already added — skipping")

open("templates/index.html", "w").write(idx_content)
print("\nDone.")
