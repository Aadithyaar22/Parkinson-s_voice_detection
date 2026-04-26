"""
Extract wav2vec2-XLS-R embeddings from a folder of audio files.

Why XLS-R: pretrained on 128 languages including Hindi, Tamil, Telugu,
Bengali, Urdu, Marathi. Produces language-robust speech representations.
We freeze the model and use it as a feature extractor; we do NOT
fine-tune it (that needs real GPU).

Output: a CSV with columns
    name, subject, status, emb_0000, emb_0001, ..., emb_1023, error
One row per audio file.

Intended workflows:

  1. Extract from the Italian training corpus:
     python scripts/extract_wav2vec2_embeddings.py \
         --input  ~/Downloads/ItalianPVS \
         --output italian_w2v2.csv \
         --labels auto

  2. Extract from YOUR recordings for a sanity check:
     python scripts/extract_wav2vec2_embeddings.py \
         --input  ~/Desktop/my_recordings \
         --output my_recordings_w2v2.csv \
         --labels unknown       # files aren't labelled, status will be -1

Runtime on M4 Air with MPS: ~1-2 seconds per recording after warmup.
Expect ~15-20 minutes for the full Italian dataset.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Add project root so `from src.*` can resolve if user runs from elsewhere
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

# We import torch + transformers inside main() so the --help message loads
# fast even on a cold conda env.

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}


# ---------------------------------------------------------------------------
# Label inference (same logic as extract_features_from_audio.py)
# ---------------------------------------------------------------------------
_PD_KEYWORDS = [r"\bpd\b", r"parkinson", r"pwpd", r"people with parkinson",
                r"patient", r"sick", r"disease"]
_HC_KEYWORDS = [r"\bhc\b", r"healthy", r"control", r"normal"]
_GROUP_FOLDER_PATTERNS = [
    r"healthy", r"control", r"\bhc\b", r"\bpd\b", r"parkinson",
    r"pwpd", r"patient", r"disease", r"elderly", r"young",
    r"people with", r"normal", r"dataset", r"audio", r"recordings?",
]


def infer_label(path: Path) -> Optional[int]:
    s = str(path).lower()
    pd_match = any(re.search(p, s) for p in _PD_KEYWORDS)
    hc_match = any(re.search(p, s) for p in _HC_KEYWORDS)
    if hc_match and not pd_match:
        return 0
    if pd_match and not hc_match:
        return 1
    parts = path.parts
    for part in reversed(parts):
        p = part.lower()
        if any(re.search(k, p) for k in _HC_KEYWORDS):
            return 0
        if any(re.search(k, p) for k in _PD_KEYWORDS):
            return 1
    return None


def _looks_like_group_folder(name: str) -> bool:
    n = name.lower()
    return any(re.search(p, n) for p in _GROUP_FOLDER_PATTERNS)


def infer_subject(path: Path, root: Path) -> str:
    try:
        rel_parents = list(path.relative_to(root).parent.parts)
    except ValueError:
        rel_parents = list(path.parent.parts[-3:])
    for folder in reversed(rel_parents):
        if folder in ("", ".", "/"):
            continue
        if _looks_like_group_folder(folder):
            continue
        return folder
    return path.stem


# ---------------------------------------------------------------------------
# File walker
# ---------------------------------------------------------------------------
def collect_files(root: Path, label_mode: str) -> List[Tuple[Path, str, int, str]]:
    """Return list of (path, subject, label, rel_path).
    label_mode: 'auto' | 'unknown'
    In 'unknown', label=-1 and subject defaults to filename stem.
    """
    root = root.resolve()
    items: List[Tuple[Path, str, int, str]] = []
    skipped: List[str] = []

    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue
        rel = str(p.relative_to(root))

        if label_mode == "unknown":
            label = -1
            subject = p.stem
        else:  # 'auto'
            label = infer_label(p)
            subject = infer_subject(p, root)
            if label is None:
                skipped.append(rel)
                continue

        items.append((p, subject, label, rel))

    if skipped:
        print(f"[warn] {len(skipped)} files skipped (could not infer label). "
              f"Examples:", file=sys.stderr)
        for s in skipped[:5]:
            print(f"    {s}", file=sys.stderr)
        print("[warn] Use --labels unknown to keep them with status=-1.",
              file=sys.stderr)

    return items


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """wav2vec2 expects 16 kHz mono float32 in [-1, 1]."""
    import librosa
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    # Trim leading/trailing silence, same as the feature extractor
    y, _ = librosa.effects.trim(y, top_db=25)
    if y.size < target_sr // 4:  # shorter than 0.25s
        # Pad so the model doesn't choke
        y = np.pad(y, (0, target_sr // 4 - y.size))
    # Peak normalise
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def get_device(force_cpu: bool = False):
    import torch
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_embeddings(
    items: List[Tuple[Path, str, int, str]],
    output_csv: Path,
    model_name: str = "facebook/wav2vec2-xls-r-300m",
    force_cpu: bool = False,
    max_seconds: float = 12.0,
) -> None:
    """
    Pump audio through a frozen wav2vec2 model, mean-pool across time,
    write one CSV row per input.

    max_seconds: clip audio to this length. Long clips blow up memory
    because the transformer is O(T^2) in sequence length. 12s of speech
    is plenty for sustained vowels.
    """
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    device = get_device(force_cpu=force_cpu)
    print(f"[info] device: {device}")
    print(f"[info] loading model (first run downloads ~1.2GB): {model_name}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    # Peek at the hidden size so we know how many emb_* columns to write
    hidden_size = model.config.hidden_size
    print(f"[info] embedding dim: {hidden_size}")

    fieldnames = (
        ["name", "subject", "status"]
        + [f"emb_{i:04d}" for i in range(hidden_size)]
        + ["error"]
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output_csv, "w", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()
    out_f.flush()

    t0 = time.time()
    done = 0
    failures = 0
    total = len(items)

    max_samples = int(max_seconds * 16000)

    try:
        with torch.no_grad():
            for path, subject, label, rel in items:
                row = {"name": rel, "subject": subject, "status": label,
                       "error": None}
                try:
                    y = load_audio(str(path))
                    if y.size > max_samples:
                        y = y[:max_samples]
                    inputs = feature_extractor(
                        y, sampling_rate=16000, return_tensors="pt",
                        padding=False,
                    )
                    input_values = inputs.input_values.to(device)
                    # Forward pass - we take the last hidden state and
                    # mean-pool across the time axis.
                    out = model(input_values)
                    # out.last_hidden_state: (1, T, hidden_size)
                    emb = out.last_hidden_state.mean(dim=1).squeeze(0)
                    emb = emb.detach().cpu().numpy().astype(np.float32)
                    for i, v in enumerate(emb):
                        row[f"emb_{i:04d}"] = float(v)
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"
                    failures += 1

                writer.writerow(row)
                out_f.flush()
                done += 1
                _log_progress(done, total, t0, failures)
    finally:
        out_f.close()

    print(f"\n[done] wrote {output_csv}  ({done} rows, {failures} failed)")


def _log_progress(done: int, total: int, t0: float, failures: int) -> None:
    dt = time.time() - t0
    rate = done / dt if dt > 0 else 0
    eta = (total - done) / rate if rate > 0 else float("inf")
    bar_len = 30
    filled = int(bar_len * done / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stderr.write(
        f"\r[{bar}] {done}/{total}  {rate:.2f} file/s  "
        f"ETA {eta/60:5.1f} min  failed={failures}"
    )
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="Folder of audio files (walked recursively)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Destination CSV path")
    ap.add_argument("--labels", choices=["auto", "unknown"], default="auto",
                    help="'auto' infers labels from path, "
                         "'unknown' labels everything -1 (for your own test recordings)")
    ap.add_argument("--model", default="facebook/wav2vec2-xls-r-300m",
                    help="HuggingFace model ID. xls-r-300m is multilingual, "
                         "including Indian languages.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N files (useful for smoke testing)")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU (default: auto-detect MPS/CUDA/CPU)")
    ap.add_argument("--max-seconds", type=float, default=12.0,
                    help="Clip audio longer than this (default 12s; transformer "
                         "is O(T^2) so long files eat memory)")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"Input folder does not exist: {args.input}")

    print(f"[info] scanning {args.input} ...")
    items = collect_files(args.input, args.labels)
    if args.limit:
        items = items[: args.limit]

    if not items:
        sys.exit("No audio files found.")

    # Summary + subject preview
    if args.labels == "auto":
        pd_n = sum(1 for _, _, l, _ in items if l == 1)
        hc_n = sum(1 for _, _, l, _ in items if l == 0)
        print(f"[info] {len(items)} audio files: {pd_n} PD / {hc_n} HC, "
              f"{len(set(s for _, s, _, _ in items))} subjects")

        subs = {}
        lbls = {}
        for _, s, l, _ in items:
            subs[s] = subs.get(s, 0) + 1
            lbls[s] = l
        print("[info] subject preview (first 10):")
        for s, n in list(subs.items())[:10]:
            tag = "PD" if lbls[s] == 1 else "HC"
            print(f"    [{tag}] {s!r}  ({n} recordings)")
        if len(subs) > 10:
            print(f"    ... and {len(subs) - 10} more")
        print("[info] if these look like group names rather than people, "
              "stop and fix labels.")
    else:
        print(f"[info] {len(items)} audio files (labels=unknown)")

    extract_embeddings(
        items, args.output,
        model_name=args.model,
        force_cpu=args.cpu,
        max_seconds=args.max_seconds,
    )

    print("\n[next] Upload the output CSV back to Claude for training.")


if __name__ == "__main__":
    main()
