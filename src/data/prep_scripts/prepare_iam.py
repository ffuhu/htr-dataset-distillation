#!/usr/bin/env python3
"""
Builds the parquet files of the RWTH/Aachen split of IAM from:
  - ascii/lines.txt        (transcriptions)
  - lines/...png           (line images)
  - *set*.txt              (official split lists)

Output: datasets/IAM/parquet/{train,val_1,val_2,test}.parquet
Columns: image (datasets.Image type, ready for .convert("RGB")) + transcription

The 'image' column is cast to datasets.Image() when writing, so that when the parquet
files are reloaded the image already comes out as a PIL object (no later cast step
needed).

Usage (from the repository root): python src/data/prep_scripts/prepare_iam.py <iam_dir>
  <iam_dir> contains ascii/lines.txt, lines/ and trainset.txt, validationset1.txt,
  validationset2.txt, testset.txt.
"""

import io
import sys
from pathlib import Path

from PIL import Image as PILImage
from datasets import Dataset, Features, Image, Value

# ------------------------------------------------------------------ #
# CONFIG
# ------------------------------------------------------------------ #
if len(sys.argv) != 2:
    sys.exit("uso: python src/data/prep_scripts/prepare_iam.py <iam_dir>")
BASE      = Path(sys.argv[1])
ASCII_TXT = BASE / "ascii" / "lines.txt"
LINES_DIR = BASE / "lines"

SPLIT_FILES = {
    "train": BASE / "trainset.txt",
    "val_1": BASE / "validationset1.txt",
    "val_2": BASE / "validationset2.txt",
    "test":  BASE / "testset.txt",
}

OUT_DIR = Path("datasets/IAM/parquet")

# Expected counts of the RWTH split (checksum)
EXPECTED = {"train": 6161, "val_1": 900, "val_2": 940, "test": 1861}

# ------------------------------------------------------------------ #
# 1. Parse lines.txt -> {id: transcription}
# ------------------------------------------------------------------ #
def parse_transcriptions(path):
    trans = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(" ", 8)  # 8 fixed fields + transcription
            if len(parts) < 9:
                continue
            line_id = parts[0]
            text = parts[8].replace("|", " ")  # IAM tokens -> spaces
            trans[line_id] = text
    return trans

# ------------------------------------------------------------------ #
# 2. ID -> PNG path (e04-127-04 -> lines/e04/e04-127/e04-127-04.png)
# ------------------------------------------------------------------ #
def id_to_path(line_id):
    a = line_id.split("-")[0]
    b = "-".join(line_id.split("-")[:2])
    return LINES_DIR / a / b / f"{line_id}.png"

def load_image_bytes(path):
    with PILImage.open(path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

# ------------------------------------------------------------------ #
# 3. Build each split and write parquet with the typed Image column
# ------------------------------------------------------------------ #
def build_split(name, list_path, trans):
    with open(list_path) as f:
        ids = [l.strip() for l in f if l.strip()]

    images, texts = [], []
    missing_img, missing_txt = [], []
    for lid in ids:
        if lid not in trans:
            missing_txt.append(lid)
            continue
        img_path = id_to_path(lid)
        if not img_path.exists():
            missing_img.append(lid)
            continue
        images.append({"bytes": load_image_bytes(img_path), "path": f"{lid}.png"})
        texts.append(trans[lid])

    features = Features({"image": Image(), "transcription": Value("string")})
    ds = Dataset.from_dict({"image": images, "transcription": texts}, features=features)

    print(f"[{name}] filas={len(ds)} esperado={EXPECTED[name]} "
          f"| sin_txt={len(missing_txt)} sin_img={len(missing_img)}")
    if missing_txt[:3]:
        print(f"   ej. sin transcripcion: {missing_txt[:3]}")
    if missing_img[:3]:
        print(f"   ej. sin imagen: {missing_img[:3]}")
    return ds

def main():
    assert ASCII_TXT.exists(), f"No existe {ASCII_TXT}"
    assert LINES_DIR.exists(), f"No existe {LINES_DIR}"

    print("Parseando transcripciones...")
    trans = parse_transcriptions(ASCII_TXT)
    print(f"  {len(trans)} transcripciones cargadas\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, list_path in SPLIT_FILES.items():
        if not list_path.exists():
            print(f"[{name}] AVISO: no existe la lista {list_path}, lo salto")
            continue
        ds = build_split(name, list_path, trans)
        out = OUT_DIR / f"{name}.parquet"
        ds.to_parquet(str(out))  # keeps the 'image' column as Image type
        print(f"   -> escrito {out}\n")

    print("Hecho. Los parquet ya traen la columna 'image' tipada (PIL al recargar).")

if __name__ == "__main__":
    main()
