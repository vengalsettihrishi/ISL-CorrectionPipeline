"""
prepare_training_data.py
========================
Converts processed_full_dataset/ (NPZ format) into the format
expected by train.py:

  data_iSign/
    iSign_v1.1.csv          ← video_id + english columns
    poses/
      <uid>.npy             ← (T, 225) float32 landmarks (vectors_225)
    norm_stats.npz          ← pre-computed from npz features_450

Run once before training:
    python prepare_training_data.py
"""

import csv
import numpy as np
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────
MANIFEST   = Path("processed_full_dataset/manifest.csv")
NPZ_DIRS   = {
    "train": Path("processed_full_dataset/train"),
    "val":   Path("processed_full_dataset/val"),
    "test":  Path("processed_full_dataset/test"),
}
OUT_DIR    = Path("data_iSign")
POSE_DIR   = OUT_DIR / "poses"
OUT_CSV    = OUT_DIR / "iSign_v1.1.csv"
NORM_PATH  = OUT_DIR / "norm_stats.npz"

OUT_DIR.mkdir(parents=True, exist_ok=True)
POSE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Read manifest ──────────────────────────────────────────────────────────
print("Reading manifest.csv ...")
rows = []
with open(MANIFEST, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
print(f"  {len(rows)} entries found")

# Detect split dirs
available_splits = [s for s in NPZ_DIRS if NPZ_DIRS[s].exists()]
print(f"  Available split dirs: {available_splits}")

# ─── Extract .npy files & collect features for norm stats ──────────────────
print("\nExtracting .npy landmark files ...")
good_rows = []
all_features_450 = []   # collect only train features for norm stats
skipped = 0

for row in rows:
    uid   = row["uid"]
    split = row.get("split", "train")
    text  = row.get("text", "").strip()

    if not text:
        skipped += 1
        continue

    # Locate the npz file
    npz_path = NPZ_DIRS.get(split, NPZ_DIRS["train"]) / f"{uid}.npz"
    if not npz_path.exists():
        skipped += 1
        continue

    # Load and save vectors_225 as the .npy pose file
    data = np.load(str(npz_path))

    if "vectors_225" not in data:
        print(f"  WARNING: {uid} has no 'vectors_225' key, skipping")
        skipped += 1
        continue

    landmarks = data["vectors_225"]          # shape (T, 225)
    out_npy = POSE_DIR / f"{uid}.npy"
    np.save(str(out_npy), landmarks)

    good_rows.append({"video_id": uid, "english": text})

    # Collect features_450 from training split only for norm stats
    if split == "train" and "features_450" in data:
        all_features_450.append(data["features_450"])  # (T, 450)

print(f"  Extracted {len(good_rows)} samples  ({skipped} skipped)")

# ─── Write compatible CSV ───────────────────────────────────────────────────
print(f"\nWriting {OUT_CSV} ...")
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["video_id", "english"])
    writer.writeheader()
    writer.writerows(good_rows)
print(f"  Done — {len(good_rows)} rows")

# ─── Compute & save norm stats from training set ────────────────────────────
if all_features_450:
    print(f"\nComputing norm stats from {len(all_features_450)} training samples ...")
    stacked = np.concatenate(all_features_450, axis=0)   # (N_total_frames, 450)
    mean = stacked.mean(axis=0).astype(np.float32)
    std  = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0   # prevent division by zero
    np.savez(str(NORM_PATH), mean=mean, std=std)
    print(f"  Saved norm_stats.npz  (mean shape {mean.shape}, std shape {std.shape})")
else:
    print("  WARNING: no training features found, norm_stats.npz NOT created")

print("\n✅ Done! You can now run:  python train.py")
print(f"   CSV:        {OUT_CSV}")
print(f"   Poses dir:  {POSE_DIR}  ({len(good_rows)} .npy files)")
print(f"   Norm stats: {NORM_PATH}")
