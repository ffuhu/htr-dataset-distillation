"""
Prepares the READ2016 dataset (Ratsprotokolle, ICFHR2016 HTR competition, Zenodo
10.5281/zenodo.1164045 -- open access, no registration) from the archives
Train-And-Val-ICFHR-2016.tgz and Test-ICFHR-2016.tgz, downloaded and extracted into
datasets/READ2016/raw/.

Standard PAGE XML format (no special tokens, unlike GW): each <TextLine> carries its
<Coords> polygon and its literal transcription in <TextEquiv><Unicode>. Each line is cropped
from the full page image using the bounding box of its polygon.

Splits: Training/Validation come separated in the dataset itself (PublicData/Training,
PublicData/Validation); the test set is the competition one (Test-ICFHR-2016/).

Usage: python prepare_read2016.py <raw_dir> <output_dir>
Writes <output_dir>/parquet/{train,val,test}.parquet with columns "image" and "transcription".
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from datasets import Dataset, Image as HFImage

PAGE_NS = {"pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}


def parse_page_xml(xml_path: Path) -> tuple[str, list[tuple[tuple[int, int, int, int], str]]]:
	tree = ET.parse(xml_path)
	root = tree.getroot()
	page = root.find("pc:Page", PAGE_NS)
	image_filename = page.get("imageFilename")

	lines = []
	for text_line in page.findall(".//pc:TextLine", PAGE_NS):
		coords = text_line.find("pc:Coords", PAGE_NS)
		points_str = coords.get("points", "")
		pts = [tuple(map(int, p.split(","))) for p in points_str.split() if p]
		if not pts:
			continue
		xs = [p[0] for p in pts]
		ys = [p[1] for p in pts]
		bbox = (min(xs), min(ys), max(xs), max(ys))

		unicode_el = text_line.find("pc:TextEquiv/pc:Unicode", PAGE_NS)
		text = unicode_el.text if unicode_el is not None and unicode_el.text else None
		if text is None or text.strip() == "":
			continue  # no transcription -- not usable for supervised HTR
		if len(text) <= 1:
			# `CRNN_CTC_Wrapper.test_step` used to compute `Y[idx, :L[idx]-1]` (asymmetric with
			# validation_step, which does not subtract 1) -- with a transcription of length 1 the
			# test target became empty and `SER()` raised ZeroDivisionError (found with READ2016).
			# Not specific to this dataset -- an asymmetry in the shared code that any dataset with
			# one-character lines can trigger. These lines were excluded here as a contained, local
			# fix. test_step has since been fixed; the exclusion is kept so the data match the paper.
			continue

		lines.append((bbox, text))

	return image_filename, lines


def process_split(page_dir: Path, images_dir: Path, out_dir: Path, split_name: str) -> list[dict]:
	out_dir.mkdir(parents=True, exist_ok=True)
	samples = []

	xml_files = sorted(page_dir.glob("*.xml"))
	print(f"[{split_name}] {len(xml_files)} paginas")

	for xml_path in xml_files:
		image_filename, lines = parse_page_xml(xml_path)
		image_path = images_dir / image_filename
		if not image_path.exists():
			print(f"  [WARN] falta imagen {image_path}, saltando pagina")
			continue

		page_img = Image.open(image_path).convert("L")
		page_id = xml_path.stem

		for i, (bbox, text) in enumerate(lines):
			crop = page_img.crop(bbox)
			if crop.width < 2 or crop.height < 2:
				continue
			out_path = out_dir / f"{page_id}-{i:03d}.png"
			crop.save(out_path)
			samples.append({"image": str(out_path), "transcription": text})

	return samples


def main(raw_dir: str, output_dir: str):
	raw = Path(raw_dir)
	out = Path(output_dir)

	splits = {
			"train": (raw / "PublicData/Training/page", raw / "PublicData/Training/Images"),
			"val": (raw / "PublicData/Validation/page", raw / "PublicData/Validation/Images"),
			"test": (raw / "Test-ICFHR-2016/page", raw / "Test-ICFHR-2016"),
			}

	parquet_dir = out / "parquet"
	parquet_dir.mkdir(parents=True, exist_ok=True)

	for split_name, (page_dir, images_dir) in splits.items():
		samples = process_split(page_dir, images_dir, out / "line_images" / split_name, split_name)
		print(f"  {split_name}: {len(samples)} lineas")
		ds = Dataset.from_list(samples)
		ds = ds.cast_column("image", HFImage())
		ds.to_parquet(str(parquet_dir / f"{split_name}.parquet"))

	print(f"Done. Parquet files in {parquet_dir}")


if __name__ == "__main__":
	main(
			sys.argv[1] if len(sys.argv) > 1 else "datasets/READ2016/raw",
			sys.argv[2] if len(sys.argv) > 2 else "datasets/READ2016",
			)
