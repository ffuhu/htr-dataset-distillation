"""
Prepares the George Washington (GW) dataset from the partial GitHub mirror
(hinderling/keyword_spotting) -- the official download from fki.tic.heia-fr.ch requires a
registered account (FKI_USER/FKI_PASSWORD, as in the download.sh of PyLaia-examples). This
mirror covers 15 of the 20 official GW pages (270-279, 300-304 -- 305-309 are missing), at
WORD level (full page image + SVG polygon per word + character-by-character transcription
with special tokens such as `s_pt`). It rebuilds LINES (not words) by grouping on the
page-line prefix of the ID ("270-01-*"), using the union of the bounding boxes of the words
of each line to crop from the full page image -- more faithful than pasting the word crops
together.

Special token table (source: legacy/washington/src/fki/prepare.sh of
carmocca/PyLaia-examples -- production script of the same PRHLT group that created
IAM-HistDB, not our own assumption).

Usage: python prepare_gw.py <output_dir>
Writes <output_dir>/parquet/{train,val,test}.parquet with columns "image" (path to PNG) and
"transcription" (str), ready for `configs/data/prep/htr_lines.yaml` + `data.TranscriptionCTC.HFCTC`.
"""
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from datasets import Dataset, Image as HFImage

RAW_BASE = "https://raw.githubusercontent.com/hinderling/keyword_spotting/master/data"
PAGES = [270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 300, 301, 302, 303, 304]

SPECIAL_TOKENS = {
	"s_bl": "(", "s_br": ")", "s_cm": ",", "s_et": "&", "s_lb": "£",
	"s_mi": "-", "s_pt": ".", "s_qo": ":", "s_qt": "'", "s_s": "s",
	"s_sl": "/", "s_sq": ";", "s_GW": "G.W.",
	}

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}
NUM_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

TEST_PAGES = {303, 304}
VAL_PAGES = {301, 302}


def decode_token(tok: str) -> str:
	if tok in SPECIAL_TOKENS:
		return SPECIAL_TOKENS[tok]
	if tok.startswith("s_"):
		return tok[2:]  # e.g. s_1755 -> "1755", s_1st -> "1st" (spelled out, matches the handwriting)
	return tok


def decode_word(tokens: list[str]) -> str:
	return "".join(decode_token(t) for t in tokens)


def fetch(url: str, dest: Path):
	dest.parent.mkdir(parents=True, exist_ok=True)
	if not dest.exists():
		urllib.request.urlretrieve(url, dest)


def parse_svg_bboxes(svg_path: Path) -> dict[str, tuple[float, float, float, float]]:
	tree = ET.parse(svg_path)
	bboxes = {}
	for path in tree.getroot().findall(".//svg:path", SVG_NS):
		word_id = path.get("id")
		d = path.get("d", "")
		nums = [float(n) for n in NUM_RE.findall(d)]
		xs = nums[0::2]
		ys = nums[1::2]
		if xs and ys:
			bboxes[word_id] = (min(xs), min(ys), max(xs), max(ys))
	return bboxes


def main(output_dir: str):
	raw_dir = Path(output_dir) / "raw"
	lines_dir = Path(output_dir) / "line_images"
	lines_dir.mkdir(parents=True, exist_ok=True)

	print("Downloading transcription.txt...")
	transcription_path = raw_dir / "transcription.txt"
	fetch(f"{RAW_BASE}/ground-truth/transcription.txt", transcription_path)

	word_tokens: dict[str, list[str]] = {}
	for line in transcription_path.read_text().splitlines():
		if not line.strip():
			continue
		word_id, toks = line.split(" ", 1)
		word_tokens[word_id] = toks.split("-")

	all_samples = []  # (page, line_key, image_path, transcription)

	for page in PAGES:
		print(f"Processing page {page}...")
		img_path = raw_dir / "images" / f"{page}.jpg"
		svg_path = raw_dir / "locations" / f"{page}.svg"
		fetch(f"{RAW_BASE}/images/{page}.jpg", img_path)
		fetch(f"{RAW_BASE}/ground-truth/locations/{page}.svg", svg_path)

		bboxes = parse_svg_bboxes(svg_path)
		page_img = Image.open(img_path).convert("L")

		lines: dict[str, list[str]] = {}  # "270-01" -> [word_id, ...] in word order
		for word_id in word_tokens:
			if not word_id.startswith(f"{page}-"):
				continue
			if word_id not in bboxes:
				continue  # word without a location entry -- skip, don't fabricate a box
			_, line_no, word_no = word_id.split("-")
			line_key = f"{page}-{line_no}"
			lines.setdefault(line_key, []).append(word_id)

		for line_key, word_ids in lines.items():
			word_ids = sorted(word_ids, key=lambda w: int(w.split("-")[-1]))
			boxes = [bboxes[w] for w in word_ids]
			x0 = min(b[0] for b in boxes)
			y0 = min(b[1] for b in boxes)
			x1 = max(b[2] for b in boxes)
			y1 = max(b[3] for b in boxes)

			crop = page_img.crop((x0, y0, x1, y1))
			out_path = lines_dir / f"{line_key}.png"
			crop.save(out_path)

			transcription = " ".join(decode_word(word_tokens[w]) for w in word_ids)
			all_samples.append((page, line_key, str(out_path), transcription))

	print(f"Total lines reconstructed: {len(all_samples)}")

	splits: dict[str, list] = {"train": [], "val": [], "test": []}
	for page, line_key, image_path, transcription in all_samples:
		split = "test" if page in TEST_PAGES else "val" if page in VAL_PAGES else "train"
		splits[split].append({"image": image_path, "transcription": transcription})

	for split, samples in splits.items():
		print(f"  {split}: {len(samples)} lines")

	parquet_dir = Path(output_dir) / "parquet"
	parquet_dir.mkdir(parents=True, exist_ok=True)
	for split, samples in splits.items():
		ds = Dataset.from_list(samples)
		ds = ds.cast_column("image", HFImage())
		ds.to_parquet(str(parquet_dir / f"{split}.parquet"))

	print(f"Done. Parquet files in {parquet_dir}")


if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "datasets/GW")
