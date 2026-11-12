"""Clean and split ORIGINAL potato images before any random augmentation.

Place the three Potato___* folders inside data/raw/ (not augmented copies).
Run: python prepare_data.py
Repeat with: python prepare_data.py --force
"""
import argparse
import hashlib
import random
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
DEST = ROOT / "data" / "split"
CLASSES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare(force: bool = False) -> dict:
    if not all((RAW / name).is_dir() for name in CLASSES):
        raise FileNotFoundError(
            "Missing class folders. Copy Potato___Early_blight, "
            "Potato___Late_blight and Potato___healthy into data/raw/. "
            "See DATASET_GUIDE.md."
        )
    if DEST.exists() and any(DEST.rglob("*.jpg")):
        if not force:
            print("Split data already exists. Skipping. To rebuild, use --force.")
            return {}
        shutil.rmtree(DEST)
    # Determine data integrity before writing any split.
    rng = random.Random(42)
    summary = {}
    seen_hashes = set()
    for class_name in CLASSES:
        valid = []
        bad = 0
        duplicates = 0
        for path in sorted((RAW / class_name).iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                if file_hash in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes.add(file_hash)
                valid.append(path)
            except (UnidentifiedImageError, OSError, ValueError):
                bad += 1
        if len(valid) < 20:
            raise ValueError(f"{class_name}: need at least 20 valid images; found {len(valid)}")
        rng.shuffle(valid)
        n = len(valid)
        n_train = int(n * .70)
        n_val = int(n * .15)
        partitions = {
            "train": valid[:n_train],
            "val": valid[n_train:n_train + n_val],
            "test": valid[n_train + n_val:]
        }
        for split_name, source_paths in partitions.items():
            target_dir = DEST / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for number, source_path in enumerate(source_paths):
                # PIL always gives RGB; OpenCV's BGR conversion is not needed.
                with Image.open(source_path) as img:
                    img.convert("RGB").save(target_dir / f"{number:05d}.jpg", quality=94)
        summary[class_name] = {k: len(v) for k, v in partitions.items()}
        print(f"{class_name}: {summary[class_name]} (invalid {bad}, duplicate {duplicates})")
    print("Done: data/split/train, data/split/val, data/split/test")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerate data/split only")
    args = parser.parse_args()
    prepare(force=args.force)
