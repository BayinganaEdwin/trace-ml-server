"""
prepare_dataset.py — Download & organise construction images for training
=========================================================================

Usage
-----
  # Primary: Roboflow (needs free API key from roboflow.com/settings/api)
  export ROBOFLOW_API_KEY=your_key
  python prepare_dataset.py --roboflow --augment --min-per-class 80

  # Fallback: Bing image scraper (no API key needed)
  pip install icrawler
  python prepare_dataset.py --scrape --augment --min-per-class 80

  # Both together
  python prepare_dataset.py --roboflow --scrape --augment --min-per-class 80
"""

import os
import sys
import shutil
import pathlib
import argparse
import random
import yaml

DATA_DIR = pathlib.Path("data")
RAW_DIR  = pathlib.Path("raw_downloads")

CLASS_NAMES = [
    "formwork_erected",
    "reinforcement_placed",
    "concrete_poured",
    "structure_complete",
]

# ── Roboflow source class → our target class ──────────────────────────────────
# Covers both datasets + common alternates
CLASS_MAPPING = {
    # ntucae/m.-rebar
    "formwork_local":            "formwork_erected",
    "rebar_local":               "reinforcement_placed",
    "concrete_local":            "concrete_poured",
    # lab-g2wzu/c1-8dwv2
    "i_placing_formwork":        "formwork_erected",
    "i_placing/fixing_rebar":    "reinforcement_placed",
    "i_pouring_concrete":        "concrete_poured",
    "placing_formwork":          "formwork_erected",
    "placing_rebar":             "reinforcement_placed",
    "pouring_concrete":          "concrete_poured",
    "fixing_rebar":              "reinforcement_placed",
    # Generic names that might appear
    "formwork":                  "formwork_erected",
    "rebar":                     "reinforcement_placed",
    "reinforcement":             "reinforcement_placed",
    "concrete":                  "concrete_poured",
    "structure":                 "structure_complete",
    # Already mapped
    "formwork_erected":          "formwork_erected",
    "reinforcement_placed":      "reinforcement_placed",
    "concrete_poured":           "concrete_poured",
    "structure_complete":        "structure_complete",
}

# Roboflow datasets to try  (workspace, project, version, format)
# Using "yolov8" — works for both classification AND object-detection projects
ROBOFLOW_DATASETS = [
    ("ntucae",    "m.-rebar",   1, "yolov8"),
    ("lab-g2wzu", "c1-8dwv2",  1, "yolov8"),
]

for cls in CLASS_NAMES:
    (DATA_DIR / cls).mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)


# ── Parse a YOLOv8 downloaded folder ─────────────────────────────────────────
def extract_from_yolov8_folder(folder: pathlib.Path):
    """
    A YOLOv8 export has:
      folder/
        data.yaml          ← maps index → class name
        train/images/      ← images
        train/labels/      ← .txt files: one line per box: <cls_idx> x y w h
        valid/images/
        valid/labels/
        test/images/       (optional)
        test/labels/       (optional)

    Strategy: for each image, read its label file → get cls_idx → map to our class.
    If no label file exists, skip (can't know the class).
    """
    yaml_path = folder / "data.yaml"
    if not yaml_path.exists():
        print(f"    No data.yaml found in {folder}, skipping")
        return 0

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    # Build index → roboflow class name map
    names = meta.get("names", {})
    if isinstance(names, list):
        idx_to_name = {i: n for i, n in enumerate(names)}
    elif isinstance(names, dict):
        idx_to_name = {int(k): v for k, v in names.items()}
    else:
        print(f"    Unexpected names format in data.yaml: {names}")
        return 0

    print(f"    Classes in dataset: {list(idx_to_name.values())}")

    copied = 0
    for split in ["train", "valid", "test"]:
        img_dir   = folder / split / "images"
        label_dir = folder / split / "labels"
        if not img_dir.exists():
            continue

        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            # Read first annotation line → class index
            lines = label_path.read_text().strip().splitlines()
            if not lines:
                continue
            try:
                cls_idx = int(lines[0].split()[0])
            except (ValueError, IndexError):
                continue

            rf_class = idx_to_name.get(cls_idx, "").lower()
            target   = CLASS_MAPPING.get(rf_class)
            if not target:
                continue

            dst = DATA_DIR / target / img_path.name
            shutil.copy2(img_path, dst)
            copied += 1

    return copied


# ── Roboflow download ─────────────────────────────────────────────────────────
def download_roboflow(api_key: str):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  roboflow package not found. Run: pip install roboflow")
        return

    rf = Roboflow(api_key=api_key)
    total = 0
    for workspace, project, version, fmt in ROBOFLOW_DATASETS:
        try:
            print(f"\n  Downloading {workspace}/{project} v{version} as {fmt}...")
            proj = rf.workspace(workspace).project(project)
            ds   = proj.version(version).download(fmt, location=str(RAW_DIR / project))
            folder = pathlib.Path(RAW_DIR / project)
            n = extract_from_yolov8_folder(folder)
            print(f"  Extracted {n} images from {workspace}/{project}")
            total += n
        except Exception as e:
            print(f"  Failed {workspace}/{project}: {e}")
    print(f"\n  Roboflow total: {total} images extracted")


# ── Bing image scraper ────────────────────────────────────────────────────────
SCRAPE_QUERIES = {
    "formwork_erected": [
        "concrete formwork construction site",
        "wooden formwork erected building",
        "shuttering formwork column slab",
        "construction formwork timber frame",
    ],
    "reinforcement_placed": [
        "rebar reinforcement steel construction site",
        "steel reinforcement placed concrete slab",
        "rebar grid construction floor",
        "reinforced concrete steel bars placed",
    ],
    "concrete_poured": [
        "concrete pouring construction site",
        "fresh concrete poured slab",
        "concrete placement construction workers",
        "wet concrete poured building structure",
    ],
    "structure_complete": [
        "completed concrete building frame structure",
        "finished reinforced concrete structure",
        "concrete building frame complete construction",
        "concrete skeleton building complete",
    ],
}

def scrape_images(min_per_class: int = 80):
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("  icrawler not found. Run: pip install icrawler")
        return

    for cls, queries in SCRAPE_QUERIES.items():
        dst      = DATA_DIR / cls
        existing = len(list(dst.glob("*.jpg")) + list(dst.glob("*.png")))
        if existing >= min_per_class:
            print(f"  {cls}: {existing} images — already at target, skipping scrape")
            continue

        needed    = min_per_class - existing
        per_query = max(5, needed // len(queries) + 1)
        print(f"\n  Scraping ~{needed} images for '{cls}'...")

        for q in queries:
            crawler = BingImageCrawler(
                storage={"root_dir": str(dst)},
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=4,
            )
            crawler.crawl(keyword=q, max_num=per_query, file_idx_offset="auto")

        final = len(list(dst.glob("*.jpg")) + list(dst.glob("*.png")))
        print(f"  {cls}: {existing} → {final} images")


# ── Augment small classes ─────────────────────────────────────────────────────
def augment_small_classes(min_per_class: int = 80):
    try:
        from PIL import Image, ImageOps, ImageEnhance
    except ImportError:
        print("  Pillow not found. Run: pip install pillow")
        return

    for cls in CLASS_NAMES:
        dst  = DATA_DIR / cls
        imgs = list(dst.glob("*.jpg")) + list(dst.glob("*.jpeg")) + list(dst.glob("*.png"))
        n    = len(imgs)
        if n == 0 or n >= min_per_class:
            continue

        need = min_per_class - n
        print(f"\n  {cls}: {n} images — generating {need} augmented copies...")
        random.shuffle(imgs)
        generated = 0
        idx = 0
        while generated < need:
            src = imgs[idx % len(imgs)]
            idx += 1
            try:
                img    = Image.open(src).convert("RGB")
                choice = random.random()
                if choice < 0.20:
                    img = ImageOps.mirror(img)
                elif choice < 0.40:
                    img = img.rotate(random.uniform(-20, 20), fillcolor=(128, 128, 128))
                elif choice < 0.55:
                    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
                elif choice < 0.70:
                    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
                elif choice < 0.85:
                    img = ImageEnhance.Saturation(img).enhance(random.uniform(0.8, 1.2))
                else:
                    # horizontal flip + brightness
                    img = ImageOps.mirror(img)
                    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))

                out = dst / f"aug_{generated:04d}_{src.stem}.jpg"
                img.save(out, "JPEG", quality=90)
                generated += 1
            except Exception as e:
                print(f"    skip {src.name}: {e}")

        final = len(list(dst.glob("*.jpg")))
        print(f"  {cls}: {n} → {final} images after augmentation")


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary():
    print("\n── Final dataset summary ──")
    total = 0
    for cls in CLASS_NAMES:
        imgs = (list((DATA_DIR / cls).glob("*.jpg"))
              + list((DATA_DIR / cls).glob("*.jpeg"))
              + list((DATA_DIR / cls).glob("*.png")))
        n = len(imgs)
        total += n
        bar = "█" * min(40, n // 2) if n > 0 else ""
        tag = " ⚠  empty" if n == 0 else (" ⚠  low" if n < 30 else "")
        print(f"  {cls:<25}  {n:>4}  {bar}{tag}")
    print(f"  {'TOTAL':<25}  {total:>4}")
    if total == 0:
        print("\n  No images found. Try: python prepare_dataset.py --scrape --augment")
    elif total < 100:
        print("\n  ⚠  Dataset is small. Run with --augment (already included above if used).")
    else:
        print("\n  ✓  Ready to train. Run: python train.py")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roboflow",      action="store_true", help="Download from Roboflow")
    parser.add_argument("--scrape",        action="store_true", help="Scrape Bing Images (needs icrawler)")
    parser.add_argument("--augment",       action="store_true", help="Augment small classes")
    parser.add_argument("--min-per-class", type=int, default=80)
    args = parser.parse_args()

    print("=" * 60)
    print("SiteLog Dataset Preparation")
    print("=" * 60)

    if args.roboflow:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            print("\n  ERROR: ROBOFLOW_API_KEY not set.")
            print("  Run: export ROBOFLOW_API_KEY=your_key_here")
            sys.exit(1)
        print("\n── Downloading from Roboflow ──")
        download_roboflow(api_key)

    if args.scrape:
        print("\n── Scraping images ──")
        scrape_images(min_per_class=args.min_per_class)

    if args.augment:
        print("\n── Augmenting small classes ──")
        augment_small_classes(min_per_class=args.min_per_class)

    if not any([args.roboflow, args.scrape, args.augment]):
        print("\nNo action specified. Use --roboflow, --scrape, or --augment.")
        print("Example: python prepare_dataset.py --scrape --augment")

    print_summary()


if __name__ == "__main__":
    main()
