"""
Batch feature extraction: walk a folder of audio files, extract the
full 56-feature set (UCI-22 + 34 extras), and write a CSV.

Run this on YOUR machine (MacBook / RTX box) on the Italian Parkinson's
dataset (or any folder of labelled audio). Upload the resulting CSV back
to Claude for cross-corpus retraining.

Usage
-----
Place this file at the project root (one level above `src/`) so the
`from src.feature_extractor import ...` import resolves, then:

    python scripts/extract_features_from_audio.py \\
        --input  /path/to/italian/dataset \\
        --output italian_features.csv \\
        --labels auto \\
        --workers 4

Label auto-detection
--------------------
If your folder is structured like:
    dataset/
      HC/{subject1,subject2,...}/{file.wav, ...}
      PD/{subject1,subject2,...}/{file.wav, ...}
or with keywords "healthy"/"parkinson" in the path, `--labels auto`
will figure it out. If that doesn't work, pass `--labels manual` and
optionally provide a `--labels-csv mapping.csv` with columns
`filename,label,subject`.

For the Italian Parkinson's Voice and Speech dataset specifically, the
published structure is:
    ItalianPVS/
      28 People with Parkinson's Disease/<subject>/<task>/*.wav
      22 Young Healthy Control/<subject>/<task>/*.wav
      15 Elderly Healthy Control/<subject>/<task>/*.wav
which `--labels auto` handles.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path so `from src.*` works even if the script is
# run from anywhere.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FEATURE_NAMES, extract_features
from src.extra_features import EXTRA_FEATURE_NAMES

ALL_FEATURE_NAMES: List[str] = list(FEATURE_NAMES) + list(EXTRA_FEATURE_NAMES)
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}


# ---------------------------------------------------------------------------
# Label inference
# ---------------------------------------------------------------------------
_PD_KEYWORDS = [
    r"\bpd\b", r"parkinson", r"pwpd", r"people with parkinson", r"patient",
    r"sick", r"disease",
]
_HC_KEYWORDS = [
    r"\bhc\b", r"healthy", r"control", r"normal",
]


def infer_label(path: Path) -> Optional[int]:
    """Return 1 for PD, 0 for healthy, None if we can't tell."""
    s = str(path).lower()
    pd_match = any(re.search(p, s) for p in _PD_KEYWORDS)
    hc_match = any(re.search(p, s) for p in _HC_KEYWORDS)
    # "Healthy Control" should win even if the parent path has "patient" etc.
    if hc_match and not pd_match:
        return 0
    if pd_match and not hc_match:
        return 1
    # Ambiguous: e.g. path contains both. Prefer the closer-to-file match.
    parts = path.parts
    for part in reversed(parts):
        p = part.lower()
        if any(re.search(k, p) for k in _HC_KEYWORDS):
            return 0
        if any(re.search(k, p) for k in _PD_KEYWORDS):
            return 1
    return None


_GROUP_FOLDER_PATTERNS = [
    r"healthy", r"control", r"\bhc\b", r"\bpd\b", r"parkinson",
    r"pwpd", r"patient", r"disease", r"elderly", r"young",
    r"people with", r"normal", r"dataset", r"audio", r"recordings?",
]


def _looks_like_group_folder(name: str) -> bool:
    n = name.lower()
    return any(re.search(p, n) for p in _GROUP_FOLDER_PATTERNS)


def infer_subject(path: Path, root: Path) -> str:
    """
    Walk from the file upward. The subject folder is the first parent
    that is NOT a group-name folder (healthy/PD/control/etc.) and NOT
    the dataset root itself. If no such folder exists (flat layout),
    use the filename stem as the subject ID.

    Examples:
      28 People with Parkinson's Disease/Mario R/B1_abc.wav  -> "Mario R"
      HC/Davide M/task1/B1.wav                               -> "Davide M"
      flat_folder/S01_file.wav                               -> stem "S01_file"
    """
    try:
        rel_parents = list(path.relative_to(root).parent.parts)
    except ValueError:
        rel_parents = list(path.parent.parts[-3:])

    # Walk from innermost folder outward
    for folder in reversed(rel_parents):
        if folder in ("", ".", "/"):
            continue
        if _looks_like_group_folder(folder):
            continue
        return folder
    return path.stem


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def process_one(args: Tuple[str, str, int, str]) -> Dict:
    """Runs in a subprocess. Extract features for one file."""
    path, subject, label, rel = args
    warnings.filterwarnings("ignore")
    try:
        feats = extract_features(
            path, compute_d2=True, compute_rpde=True, extended=True
        )
        row = {k: feats.get(k, None) for k in ALL_FEATURE_NAMES}
        row["name"] = rel
        row["subject"] = subject
        row["status"] = label
        row["error"] = None
        return row
    except Exception as e:
        row = {k: None for k in ALL_FEATURE_NAMES}
        row["name"] = rel
        row["subject"] = subject
        row["status"] = label
        row["error"] = f"{type(e).__name__}: {e}"
        return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def collect_files(root: Path, labels_csv: Optional[Path]) -> List[Tuple[str, str, int, str]]:
    """
    Return list of (absolute_path, subject_id, label_0_or_1, relative_path).
    """
    root = root.resolve()
    overrides = {}
    if labels_csv and labels_csv.exists():
        import pandas as pd
        df = pd.read_csv(labels_csv)
        for _, r in df.iterrows():
            overrides[str(r["filename"])] = {
                "label": int(r["label"]),
                "subject": str(r.get("subject", r["filename"])),
            }

    items: List[Tuple[str, str, int, str]] = []
    unlabeled: List[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        rel = str(p.relative_to(root))
        if rel in overrides:
            lab = overrides[rel]["label"]
            subj = overrides[rel]["subject"]
        else:
            lab = infer_label(p)
            subj = infer_subject(p, root)
        if lab is None:
            unlabeled.append(rel)
            continue
        items.append((str(p), subj, lab, rel))

    if unlabeled:
        print(f"[warn] {len(unlabeled)} files could not be auto-labelled and "
              f"were skipped. Example paths:", file=sys.stderr)
        for u in unlabeled[:5]:
            print(f"    {u}", file=sys.stderr)
        print("[warn] Pass --labels-csv mapping.csv to label these "
              "manually.", file=sys.stderr)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="Root folder of the audio dataset")
    ap.add_argument("--output", required=True, type=Path,
                    help="Destination CSV (e.g. italian_features.csv)")
    ap.add_argument("--labels", choices=["auto", "manual"], default="auto",
                    help="Label inference mode")
    ap.add_argument("--labels-csv", type=Path, default=None,
                    help="Optional CSV mapping filename->label[,subject] for overrides")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="Parallel worker processes (default: cpu_count - 1)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N files (useful for smoke testing)")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"Input folder does not exist: {args.input}")

    print(f"[info] scanning {args.input} ...")
    items = collect_files(args.input, args.labels_csv)
    if args.limit:
        items = items[: args.limit]
    print(f"[info] {len(items)} audio files to process, "
          f"{sum(1 for i in items if i[2]==1)} PD / {sum(1 for i in items if i[2]==0)} HC, "
          f"{len(set(i[1] for i in items))} unique subjects")

    # Show first few subjects so user can eyeball the inference
    by_subject: Dict[str, int] = {}
    subj_label: Dict[str, int] = {}
    for _, subj, lab, _ in items:
        by_subject[subj] = by_subject.get(subj, 0) + 1
        subj_label[subj] = lab
    print("[info] subject preview (first 10):")
    for subj, n in list(by_subject.items())[:10]:
        tag = "PD" if subj_label[subj] == 1 else "HC"
        print(f"    [{tag}] {subj!r}  ({n} recordings)")
    if len(by_subject) > 10:
        print(f"    ... and {len(by_subject) - 10} more")
    print("[info] if the above subjects look like group names rather than "
          "people, CV will be broken - stop now and fix labels-csv.")
    if not items:
        sys.exit("No audio files found.")

    # CSV writer setup - stream as we go so a crash doesn't lose work
    fieldnames = ["name", "subject", "status"] + ALL_FEATURE_NAMES + ["error"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output, "w", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()
    out_f.flush()

    t0 = time.time()
    done = 0
    failures = 0
    try:
        if args.workers <= 1:
            for item in items:
                row = process_one(item)
                writer.writerow(row); out_f.flush()
                done += 1
                if row["error"]: failures += 1
                _log_progress(done, len(items), t0, failures)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(process_one, it) for it in items]
                for fut in as_completed(futures):
                    row = fut.result()
                    writer.writerow(row); out_f.flush()
                    done += 1
                    if row["error"]: failures += 1
                    _log_progress(done, len(items), t0, failures)
    finally:
        out_f.close()

    print(f"\n[done] wrote {args.output}  ({done} rows, {failures} failed)")
    print("[next] Upload this CSV back to Claude and ask for retraining.")


def _log_progress(done: int, total: int, t0: float, failures: int) -> None:
    dt = time.time() - t0
    rate = done / dt if dt > 0 else 0
    eta = (total - done) / rate if rate > 0 else float("inf")
    bar_len = 30
    filled = int(bar_len * done / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stderr.write(
        f"\r[{bar}] {done}/{total}  {rate:.1f} file/s  "
        f"ETA {eta/60:5.1f} min  failed={failures}"
    )
    sys.stderr.flush()


if __name__ == "__main__":
    main()
