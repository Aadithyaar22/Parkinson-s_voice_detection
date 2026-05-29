"""
Run this from your pva2 folder:
  python patch_index.py
It adds the login button, save-reading panel, and auth JS to index.html.
"""
import re

content = open("templates/index.html").read()
changed = 0

# 1. Add mongo_available JS variable after __groq
if "window.__mongo" not in content:
    content = content.replace(
        "window.__groq = {{ groq_available|tojson }};",
        "window.__groq = {{ groq_available|tojson }};\nwindow.__mongo = {{ mongo_available|tojson }};"
    )
    changed += 1
    print("✓ Added window.__mongo")

# 2. Add login/dashboard pill in nav (after groq pill or before closing </div> of pills)
if "pvd-login-btn" not in content:
    old_pills_end = "      {% endif %}\n    </div>\n  </header>"
    new_pills_end = """      {% endif %}
      {% if mongo_available %}
        <a href="/dashboard" id="pvd-dash-link" class="pill blue" style="display:none;text-decoration:none">📊 My readings</a>
        <a href="/login" id="pvd-login-btn" class="pill" style="text-decoration:none">Sign in</a>
      {% endif %}
    </div>
  </header>"""
    if old_pills_end in content:
        content = content.replace(old_pills_end, new_pills_end)
        changed += 1
        print("✓ Added login/dashboard nav buttons")

# 3. Add save-reading panel inside result div, after groq panel (or after feat-summary)
save_panel = """
        {% if mongo_available %}
        <div class="save-panel" id="savePanel" style="display:none">
          <div class="save-header">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
            Save this reading to your history
          </div>
          <div id="save-logged-out">
            <p style="font-size:13px;color:var(--ink-mute);margin-bottom:10px">
              <a href="/login" style="color:var(--blue-600);font-weight:500">Sign in</a> to save readings and track your trend over time.
            </p>
          </div>
          <div id="save-logged-in" style="display:none">
            <textarea id="notesField" placeholder="Optional notes — e.g. 'felt tired today', 'had a cold'" rows="2"
              style="width:100%;padding:10px 12px;border:1.5px solid var(--stroke);border-radius:10px;
              font-family:var(--sans);font-size:13px;resize:vertical;margin-bottom:10px;
              color:var(--ink);background:white;outline:none;"></textarea>
            <div style="display:flex;gap:8px;align-items:center">
              <button id="saveBtn" class="btn btn-primary" style="padding:10px 20px;font-size:13px">
                💾 Save reading
              </button>
              <span id="saveMsg" style="font-size:13px;color:var(--ink-mute)"></span>
            </div>
          </div>
        </div>
        {% endif %}
"""

if "savePanel" not in content:
    # Insert before the disclaimer div
    content = content.replace(
        '        <div class="disclaimer">',
        save_panel + '\n        <div class="disclaimer">'
    )
    changed += 1
    print("✓ Added save reading panel")

# 4. Add save panel CSS before </style>
save_css = """
  .save-panel{margin-top:18px;padding:18px;border-radius:14px;
    background:var(--blue-50);border:1px solid var(--blue-100);}
  .save-header{display:flex;align-items:center;gap:8px;font-family:var(--display);
    font-weight:600;font-size:14px;color:var(--blue-700);margin-bottom:12px;}
"""
if ".save-panel" not in content:
    content = content.replace("</style>", save_css + "</style>")
    changed += 1
    print("✓ Added save panel CSS")

# 5. Add auth + save JS before </script>
auth_js = """
// ── Auth state ──────────────────────────────────────────────────────────────
function getToken() { return localStorage.getItem("pvd_token"); }

function updateAuthUI() {
  const token = getToken();
  const loginBtn = document.getElementById("pvd-login-btn");
  const dashLink = document.getElementById("pvd-dash-link");
  if (loginBtn) loginBtn.style.display = token ? "none" : "inline-flex";
  if (dashLink) dashLink.style.display = token ? "inline-flex" : "none";
  const loggedIn  = document.getElementById("save-logged-in");
  const loggedOut = document.getElementById("save-logged-out");
  if (loggedIn)  loggedIn.style.display  = token ? "block" : "none";
  if (loggedOut) loggedOut.style.display = token ? "none"  : "block";
}
updateAuthUI();

// Show save panel after analysis
const _origRender = typeof render === "function" ? render : null;
function showSavePanel() {
  const sp = document.getElementById("savePanel");
  if (sp) sp.style.display = "block";
  const sm = document.getElementById("saveMsg");
  if (sm) sm.textContent = "";
  const nb = document.getElementById("notesField");
  if (nb) nb.value = "";
}

// Save reading
const saveBtn = document.getElementById("saveBtn");
if (saveBtn) {
  saveBtn.addEventListener("click", async () => {
    if (!lastResult) return;
    const token = getToken();
    if (!token) { window.location.href = "/login"; return; }
    const notes = (document.getElementById("notesField") || {}).value || "";
    const duration = document.querySelector("audio") ? (document.querySelector("audio").duration || 0) : 0;
    saveBtn.disabled = true;
    document.getElementById("saveMsg").textContent = "Saving...";
    try {
      const res = await fetch("/api/readings", {
        method: "POST",
        headers: {"Content-Type": "application/json", "Authorization": "Bearer " + token},
        body: JSON.stringify({
          probability_pd:  lastResult.probability_pd,
          prediction:      lastResult.prediction,
          confidence_pct:  lastResult.confidence_pct,
          audio_duration_s: Math.round(duration * 10) / 10,
          backend:         lastResult.backend,
          model:           lastResult.model,
          notes:           notes,
        }),
      });
      if (res.status === 401) { localStorage.removeItem("pvd_token"); window.location.href = "/login"; return; }
      const data = await res.json();
      if (!res.ok) throw new Error(data.message);
      document.getElementById("saveMsg").textContent = "✓ Saved!";
      saveBtn.textContent = "✓ Saved";
      saveBtn.style.background = "linear-gradient(180deg,#0EA371,#059669)";
    } catch(e) {
      document.getElementById("saveMsg").textContent = "Failed: " + e.message;
      saveBtn.disabled = false;
    }
  });
}

// Hook into render to show save panel
const _renderOrig = window.render;
if (typeof _renderOrig === "function") {
  window.render = function(r) {
    _renderOrig(r);
    showSavePanel();
    updateAuthUI();
  };
}
"""

if "updateAuthUI" not in content:
    content = content.replace("</script>", auth_js + "\n</script>")
    changed += 1
    print("✓ Added auth + save JS")

open("templates/index.html", "w").write(content)
print(f"\nDone — {changed} changes applied to templates/index.html")
