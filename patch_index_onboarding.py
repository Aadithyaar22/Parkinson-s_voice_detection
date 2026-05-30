"""
Run from pva2/ folder:  python patch_index_onboarding.py
Adds 3-step onboarding modal and renames P(PD) to Parkinson's Probability.
"""

ONBOARDING_CSS = """
  /* ── Onboarding modal ───────────────────────────────────────────────────── */
  .ob-overlay{position:fixed;inset:0;z-index:1000;
    background:rgba(11,21,51,0.55);backdrop-filter:blur(6px);
    display:flex;align-items:center;justify-content:center;padding:20px;
    animation:fadeIn .3s ease;}
  @keyframes fadeIn{from{opacity:0}to{opacity:1}}
  .ob-card{background:white;border-radius:28px;padding:40px 36px 32px;
    max-width:520px;width:100%;box-shadow:0 24px 60px rgba(11,21,51,0.18);
    position:relative;animation:slideUp .4s cubic-bezier(.2,.8,.2,1);}
  @keyframes slideUp{from{transform:translateY(24px);opacity:0}to{transform:translateY(0);opacity:1}}
  .ob-step{display:none;}
  .ob-step.active{display:block;}
  .ob-icon{width:64px;height:64px;border-radius:18px;
    background:linear-gradient(135deg,#EFF6FF,#DBEAFE);
    display:flex;align-items:center;justify-content:center;
    margin-bottom:20px;font-size:28px;}
  .ob-title{font-family:"Space Grotesk",system-ui,sans-serif;font-weight:700;
    font-size:22px;letter-spacing:-0.02em;color:#0B1533;margin-bottom:10px;}
  .ob-body{font-size:14px;color:#3E4A6F;line-height:1.7;margin-bottom:24px;}
  .ob-body ul{padding-left:18px;margin-top:6px;}
  .ob-body li{margin-bottom:5px;}
  .ob-dots{display:flex;justify-content:center;gap:6px;margin-bottom:22px;}
  .ob-dot{width:7px;height:7px;border-radius:50%;background:#DBEAFE;
          transition:all .25s;}
  .ob-dot.active{background:#2563EB;width:20px;border-radius:4px;}
  .ob-actions{display:flex;justify-content:space-between;align-items:center;}
  .ob-skip{font-size:13px;color:#6B7796;cursor:pointer;background:none;
           border:none;font-family:"Inter",system-ui,sans-serif;}
  .ob-skip:hover{color:#0B1533;}
  .ob-next{font-family:"Inter",system-ui,sans-serif;font-weight:600;
    font-size:14px;padding:11px 24px;border-radius:12px;border:0;cursor:pointer;
    background:linear-gradient(180deg,#3B82F6,#1D4ED8);color:white;
    box-shadow:0 4px 14px rgba(37,99,235,0.3);}
  .ob-next:hover{transform:translateY(-1px);}
"""

ONBOARDING_HTML = """
  <!-- Onboarding modal -->
  <div class="ob-overlay" id="obOverlay" style="display:none">
    <div class="ob-card">
      <!-- Step 1 -->
      <div class="ob-step active" id="ob-1">
        <div class="ob-icon">🎙️</div>
        <div class="ob-title">How to record your voice</div>
        <div class="ob-body">
          For the most accurate result:
          <ul>
            <li>Find a <strong>quiet room</strong> — background noise affects the score</li>
            <li>Sit comfortably, hold your device <strong>10 cm from your mouth</strong></li>
            <li>Take a breath and say <strong>"aaaah"</strong> steadily for <strong>3–5 seconds</strong></li>
            <li>Keep your pitch and volume as <strong>steady as possible</strong></li>
            <li>Record at the <strong>same time each day</strong> for consistent trends</li>
          </ul>
        </div>
      </div>
      <!-- Step 2 -->
      <div class="ob-step" id="ob-2">
        <div class="ob-icon">📊</div>
        <div class="ob-title">Understanding your score</div>
        <div class="ob-body">
          After analysis you'll see a <strong>Parkinson's Probability</strong> (0–100%):
          <ul>
            <li><strong style="color:#0EA371">Below 38%</strong> — no Parkinson's indicators detected</li>
            <li><strong style="color:#DC2B4C">Above 38%</strong> — Parkinson's indicators present</li>
            <li>A <strong>single reading is not a diagnosis</strong> — trends matter more</li>
            <li>Always consult a <strong>neurologist</strong> for medical advice</li>
          </ul>
          This is a research prototype, not a medical device.
        </div>
      </div>
      <!-- Step 3 -->
      <div class="ob-step" id="ob-3">
        <div class="ob-icon">📈</div>
        <div class="ob-title">Track your progress over time</div>
        <div class="ob-body">
          <ul>
            <li><strong>Create an account</strong> to save readings and build your trend graph</li>
            <li>Record <strong>daily at the same time</strong> for consistency</li>
            <li>After <strong>6+ readings</strong> the app shows if you're improving, stable, or worsening</li>
            <li>Use <strong>Share with Doctor</strong> to send your report to your neurologist</li>
            <li>Add <strong>notes</strong> to each reading — "felt tired", "had a cold" — context matters</li>
          </ul>
        </div>
      </div>
      <!-- Progress dots -->
      <div class="ob-dots">
        <div class="ob-dot active" id="ob-dot-1"></div>
        <div class="ob-dot" id="ob-dot-2"></div>
        <div class="ob-dot" id="ob-dot-3"></div>
      </div>
      <div class="ob-actions">
        <button class="ob-skip" onclick="closeOnboarding()">Skip</button>
        <button class="ob-next" id="ob-next-btn" onclick="obNext()">Next →</button>
      </div>
    </div>
  </div>
"""

ONBOARDING_JS = """
// ── Onboarding ───────────────────────────────────────────────────────────────
let obCurrentStep = 1;
const OB_STEPS = 3;

function showOnboarding() {
  document.getElementById("obOverlay").style.display = "flex";
}
function closeOnboarding() {
  document.getElementById("obOverlay").style.display = "none";
  localStorage.setItem("pvd_onboarded", "1");
}
function obNext() {
  if (obCurrentStep >= OB_STEPS) { closeOnboarding(); return; }
  document.getElementById("ob-" + obCurrentStep).classList.remove("active");
  document.getElementById("ob-dot-" + obCurrentStep).classList.remove("active");
  obCurrentStep++;
  document.getElementById("ob-" + obCurrentStep).classList.add("active");
  document.getElementById("ob-dot-" + obCurrentStep).classList.add("active");
  if (obCurrentStep === OB_STEPS) {
    document.getElementById("ob-next-btn").textContent = "Get started ✓";
  }
}
// Show onboarding on first visit
if (!localStorage.getItem("pvd_onboarded")) {
  setTimeout(showOnboarding, 600);
}
"""

import re

path = "templates/index.html"
content = open(path).read()
changed = 0

# 1. Onboarding CSS
if ".ob-overlay" not in content:
    content = content.replace("</style>", ONBOARDING_CSS + "\n</style>")
    changed += 1
    print("✓ Added onboarding CSS")

# 2. Onboarding HTML (insert right after <body> opening or after bg divs)
if "ob-1" not in content:
    content = content.replace(
        '<div class="wrap">',
        ONBOARDING_HTML + '\n<div class="wrap">'
    )
    changed += 1
    print("✓ Added onboarding HTML")

# 3. Onboarding JS
if "showOnboarding" not in content:
    content = content.replace("</script>", ONBOARDING_JS + "\n</script>")
    changed += 1
    print("✓ Added onboarding JS")

# 4. Rename P(PD) → Parkinson's Probability in display text
# Be careful to only change display strings, not JS variable names
replacements = [
    ('P(PD) = ', 'Parkinson\'s Probability: '),
    ('>P(PD)<', '>Parkinson\'s Probability<'),
    ('"P(PD) %"', '"Parkinson\'s Probability %"'),
    ('PROBABILITY OF PARKINSON\'S', 'PARKINSON\'S PROBABILITY'),
    ('Probability of Parkinson\'s</span>', 'Parkinson\'s Probability</span>'),
    ('Probability of Parkinson\'s"', 'Parkinson\'s Probability"'),
    ('>P(PD) over time<', '>Parkinson\'s Probability over time<'),
    ('P(PD) %', 'Parkinson\'s Probability %'),
    ('P(PD) =', 'Probability:'),
]
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        changed += 1
        print(f"✓ Renamed: {old!r} → {new!r}")

open(path, "w").write(content)
print(f"\nDone — {changed} changes applied")
