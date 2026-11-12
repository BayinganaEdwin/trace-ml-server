"""
prepare_negatives.py — Download diverse non-construction images for OOD training
=================================================================================
Uses the iNaturalist public API (free, no key, returns real-world camera photos)
to download wildlife/nature observations across 10 very different taxa, giving the
'not_construction' class genuine diversity.

Also downloads TF Rock-Paper-Scissors hands dataset for human subject diversity.

Final mix (~275 images):
  iNaturalist taxa (25 each): mammals, birds, reptiles, insects, amphibians,
  fish, plants, fungi, molluscs, spiders — all clearly non-construction
  Rock-Paper-Scissors (25): human hands in various poses

Usage
-----
  cd model_training
  source venv/bin/activate
  pip install requests pillow tqdm
  python prepare_negatives.py
  python train.py   # retrain with 5 classes
"""

import io
import pathlib
import sys
import time
import zipfile
import urllib.request

try:
    import requests
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Run:\n  pip install requests pillow tqdm")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
_HERE    = pathlib.Path(__file__).parent.resolve()
DEST_DIR = _HERE / "data" / "not_construction"
TMP_DIR  = _HERE / "_neg_tmp"
IMG_SIZE = (224, 224)

# iNaturalist iconic_taxa → label prefix, 25 images each
INAT_TAXA = [
    ("Mammalia",    "mammal"),      # dogs, cats, deer, bears, foxes…
    ("Aves",        "bird"),        # pigeons, eagles, parrots, ducks…
    ("Reptilia",    "reptile"),     # lizards, snakes, turtles…
    ("Insecta",     "insect"),      # butterflies, beetles, bees…
    ("Amphibia",    "amphibian"),   # frogs, salamanders…
    ("Actinopterygii","fish"),      # trout, goldfish, bass…
    ("Plantae",     "plant"),       # trees, shrubs, leaves, grass…
    ("Fungi",       "fungus"),      # mushrooms, moulds…
    ("Mollusca",    "mollusc"),     # snails, clams, octopus…
    ("Arachnida",   "spider"),      # spiders, scorpions, crabs…
]
PER_TAXON = 25

INAT_API  = "https://api.inaturalist.org/v1/observations"
HEADERS   = {"User-Agent": "SiteLog-NegativeDataset/1.0"}

# Google TF Rock-Paper-Scissors dataset (human hands — good for diversity)
RPS_URL   = "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip"
RPS_COUNT = 25


# ── iNaturalist helpers ───────────────────────────────────────────────────────

def fetch_inat_photos(iconic_taxon: str, count: int, page: int = 1) -> list[str]:
    """
    Query iNaturalist for research-grade observations with photos.
    Returns up to `count` image URLs (medium size, ~500px).
    """
    params = {
        "photos":        "true",
        "quality_grade": "research",
        "photo_license": "cc-by,cc-by-sa,cc0,cc-by-nd",
        "iconic_taxa":   iconic_taxon,
        "per_page":      str(min(count * 2, 50)),   # fetch 2× in case some have no URL
        "page":          str(page),
        "order":         "created_at",
        "order_by":      "desc",
    }
    try:
        r = requests.get(INAT_API, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        results = r.json().get("results", [])
        urls = []
        for obs in results:
            photos = obs.get("photos", [])
            if photos:
                url = photos[0].get("url", "")
                # Replace thumbnail size 'square' (75px) with 'medium' (500px)
                url = url.replace("/square.", "/medium.")
                if url:
                    urls.append(url)
        return urls[:count]
    except Exception as e:
        print(f"    [warn] iNat query failed for {iconic_taxon}: {e}")
        return []


def download_image(url: str, dest: pathlib.Path) -> bool:
    """Download, resize to 224×224, save as JPEG."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=25)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB").resize(IMG_SIZE)
        img.save(dest, "JPEG", quality=85)
        return True
    except Exception:
        return False


# ── RPS dataset (hands) ───────────────────────────────────────────────────────

def download_rps(count: int) -> int:
    """Download Rock-Paper-Scissors training images, sample `count` of them."""
    label = "hands"
    existing = list(DEST_DIR.glob(f"{label}_*.jpg"))
    if len(existing) >= count:
        return len(existing)

    archive = TMP_DIR / "rps.zip"
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        print(f"  Downloading Rock-Paper-Scissors dataset (~85 MB) …")
        try:
            urllib.request.urlretrieve(RPS_URL, archive)
        except Exception as e:
            print(f"  [skip] RPS download failed: {e}")
            return len(existing)

    saved = len(existing)
    with zipfile.ZipFile(archive, "r") as z:
        imgs = [n for n in z.namelist()
                if n.lower().endswith(".png") and "rps-test-set" not in n]
        # Sample evenly from all 3 gesture types
        import random
        random.shuffle(imgs)
        imgs = imgs[: count * 2]

        for name in imgs:
            if saved >= count:
                break
            dest = DEST_DIR / f"{label}_{saved:04d}.jpg"
            if dest.exists():
                saved += 1
                continue
            try:
                data = z.read(name)
                img = Image.open(io.BytesIO(data)).convert("RGB").resize(IMG_SIZE)
                img.save(dest, "JPEG", quality=85)
                saved += 1
            except Exception:
                continue

    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading diverse non-construction images")
    print(f"  Source 1: iNaturalist API (wildlife/nature, 10 taxa)")
    print(f"  Source 2: TF Rock-Paper-Scissors (human hands)")
    print(f"  Destination: {DEST_DIR}")
    print("=" * 60)

    results = {}

    # ── iNaturalist taxa ──────────────────────────────────────────────────────
    for iconic_taxon, label in tqdm(INAT_TAXA, desc="iNaturalist taxa"):
        existing = list(DEST_DIR.glob(f"{label}_*.jpg"))
        if len(existing) >= PER_TAXON:
            tqdm.write(f"  ✓ {label}: already have {len(existing)}")
            results[label] = len(existing)
            continue

        tqdm.write(f"  → {label} ({iconic_taxon}) …")
        urls = fetch_inat_photos(iconic_taxon, PER_TAXON * 2)
        saved = len(existing)

        for i, url in enumerate(urls):
            if saved >= PER_TAXON:
                break
            dest = DEST_DIR / f"{label}_{saved:04d}.jpg"
            if dest.exists():
                saved += 1
                continue
            if download_image(url, dest):
                saved += 1
            time.sleep(0.05)

        tqdm.write(f"  ✓ {label}: {saved}/{PER_TAXON}")
        results[label] = saved

    # ── Rock-Paper-Scissors hands ─────────────────────────────────────────────
    tqdm.write("  → hands (Rock-Paper-Scissors) …")
    n_hands = download_rps(RPS_COUNT)
    tqdm.write(f"  ✓ hands: {n_hands}/{RPS_COUNT}")
    results["hands"] = n_hands

    # ── Summary ───────────────────────────────────────────────────────────────
    all_files = (list(DEST_DIR.glob("*.jpg")) +
                 list(DEST_DIR.glob("*.jpeg")) +
                 list(DEST_DIR.glob("*.png")))

    print(f"\n{'='*60}")
    print(f"Done. {len(all_files)} total images in {DEST_DIR}")
    print(f"{'='*60}")
    print("\nCategory breakdown:")
    for label, n in results.items():
        target = PER_TAXON if label != "hands" else RPS_COUNT
        filled = int(25 * n / target) if target else 0
        bar = "█" * filled + "░" * (25 - filled)
        print(f"  {label:<12} {bar} {n}/{target}")

    if len(all_files) == 0:
        print("\n[!] Nothing downloaded — check your internet connection.")
        sys.exit(1)

    # Clean up temp files
    import shutil
    if TMP_DIR.exists():
        ans = input("\nDelete temp archives to free space? [Y/n] ").strip().lower()
        if ans in ("", "y", "yes"):
            shutil.rmtree(TMP_DIR)
            print("  Cleaned up.")

    print("\nNext step:  python train.py")


if __name__ == "__main__":
    main()
