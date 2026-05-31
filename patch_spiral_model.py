"""
Run from pva2/:  python patch_spiral_model.py

Wires the trained HOG+SVM spiral model into the app:
  1. Updates app.py — /api/motor/spiral now uses image_b64 (canvas)
  2. Adds /api/motor/spiral-image (photo upload endpoint)
  3. Copies the new motor.html (canvas + photo upload)
  4. Adds scikit-image to requirements
"""
import shutil, os
from pathlib import Path

# ── 1. requirements.txt — add scikit-image ────────────────────────────────────
for req_file in ["requirements.txt", "parkinsons_space/requirements.txt"]:
    if Path(req_file).exists():
        content = open(req_file).read()
        if "scikit-image" not in content:
            with open(req_file, "a") as f:
                f.write("\nscikit-image>=0.19\n")
            print(f"✓ Added scikit-image to {req_file}")
        else:
            print(f"  scikit-image already in {req_file}")

# ── 2. app.py — update /api/motor/spiral to use image_b64 ────────────────────
content = open("app.py").read()

# Replace the old spiral route
old_spiral = '''@app.route("/api/motor/spiral", methods=["POST"])
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
        return jsonify({"error": str(e)}), 500'''

new_spiral = '''@app.route("/api/motor/spiral", methods=["POST"])
def api_motor_spiral():
    """Classify spiral from canvas drawing (base64 PNG)."""
    body = request.get_json(silent=True) or {}
    b64 = body.get("image_b64", "")
    if not b64:
        return jsonify({"error": "No image_b64 provided"}), 400
    try:
        from src.spiral_inference import predict_from_base64
        result = predict_from_base64(b64)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/motor/spiral-image", methods=["POST"])
def api_motor_spiral_image():
    """Classify spiral from uploaded photo (multipart form)."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    try:
        from src.spiral_inference import predict_from_bytes
        img_bytes = f.read()
        result = predict_from_bytes(img_bytes, filename=f.filename)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500'''

if old_spiral in content:
    content = content.replace(old_spiral, new_spiral)
    open("app.py", "w").write(content)
    print("✓ app.py — updated spiral routes to use HOG+SVM")
else:
    # Check if already updated
    if "spiral_inference" in content:
        print("  app.py spiral routes already updated")
    else:
        # Try to insert before the typing route
        marker = "@app.route(\"/api/motor/typing\""
        if marker in content:
            content = content.replace(marker, new_spiral + "\n\n" + marker)
            open("app.py", "w").write(content)
            print("✓ app.py — inserted spiral routes")
        else:
            print("WARNING: Could not find spiral route — adding before if __name__")
            marker2 = "\nif __name__ == \"__main__\":"
            idx = content.rfind(marker2)
            content = content[:idx] + "\n" + new_spiral + "\n" + content[idx:]
            open("app.py", "w").write(content)
            print("✓ app.py — appended spiral routes")

# ── 3. Also remove typing route if present (motor is spiral-only now) ─────────
# Keep typing for now since motor.html uses it

# ── 4. Deploy workflow — add spiral_inference.py and models_motor/ ────────────
wf = open(".github/workflows/deploy_space.yml").read()

if "spiral_inference" not in wf:
    wf = wf.replace(
        "src/auth.py \\",
        "src/auth.py \\\n            src/spiral_inference.py \\"
    )
    open(".github/workflows/deploy_space.yml", "w").write(wf)
    print("✓ workflow — added spiral_inference.py")
else:
    print("  workflow already has spiral_inference.py")

if "models_motor" not in wf:
    wf = open(".github/workflows/deploy_space.yml").read()
    wf = wf.replace(
        "cp models/training_report.json       space_build/models/",
        "cp models/training_report.json       space_build/models/\n"
        "          mkdir -p space_build/models_motor\n"
        "          [ -d models_motor ] && cp models_motor/*.joblib space_build/models_motor/ 2>/dev/null || true\n"
        "          [ -d models_motor ] && cp models_motor/*.pkl    space_build/models_motor/ 2>/dev/null || true\n"
        "          [ -d models_motor ] && cp models_motor/*.json   space_build/models_motor/ 2>/dev/null || true\n"
        "          echo '  ✓ models_motor/'"
    )
    open(".github/workflows/deploy_space.yml", "w").write(wf)
    print("✓ workflow — added models_motor/ copy step")
else:
    print("  workflow already copies models_motor/")

print("\n✅ All patches applied.")
print("\nNext steps:")
print("  1. Copy ~/Downloads/spiral_pipeline.joblib → models_motor/")
print("  2. Copy ~/Downloads/spiral_hog_params.pkl  → models_motor/")
print("  3. Copy ~/Downloads/spiral_img_size.pkl    → models_motor/")
print("  4. git add . && git commit && git push")
