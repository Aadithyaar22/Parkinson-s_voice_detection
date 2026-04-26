# Quick start

You've just unzipped the project. Here's what to do.

## To run the demo

```bash
cd pva2
pip install -r requirements.txt
pip install -r requirements_wav2vec2.txt    # ~1.5 GB, for the default wav2vec2 backend
brew install ffmpeg                          # macOS only

python app.py
```

Open http://127.0.0.1:5000 in your browser.

First click of "Analyse" downloads the wav2vec2 model (~1.2 GB from
HuggingFace — one-time). Takes 3-5 min. Subsequent requests are fast.

Click record, say "aaaah" steadily for 3-5 seconds, stop, hit Analyse.

## To re-train

See `README.md` section "Retraining", or `scripts/README.md`.

## If torch install is too painful

You don't need wav2vec2 — you can run the hand-crafted backend instead:

```bash
cp models_italian_tuned/* models/
python app.py
```

This uses the tuned voting ensemble (XGB + LGBM + RF). No torch needed,
~1.5 GB lighter install, similar accuracy (CV AUC 0.974 vs 0.972), but
predictions on non-Italian voices are less reliable.

## Project layout

See `README.md` and `FINAL_RESULTS.md`.
