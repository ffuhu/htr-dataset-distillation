"""
Prepares the LAM dataset (Ludovico Antonio Muratori, Cascianelli et al. ICPR 2022, hosted
on Kaggle -- vpippi/lam-dataset, API access with a token, CC BY 4.0) from the files
downloaded and extracted into datasets/LAM/raw/LAM/.

Unlike GW/READ2016, LAM already comes with CROPPED lines (lines/img/*.jpg) and the official
"basic" split in JSON (lines/split/basic/{train,val,test}.json) -- no need to rebuild lines
from full pages or to make up a split. Each JSON entry: {"img", "text", "width", "height",
"decade_id", "nameset"}.

Usage: python prepare_lam.py <raw_dir> <output_dir>
Writes <output_dir>/parquet/{train,val,test}.parquet with columns "image" and "transcription".
"""
import json
import sys
from pathlib import Path

from PIL import Image
from datasets import Dataset, Image as HFImage


def load_split(split_dir: Path, split_name: str) -> list[dict]:
	with open(split_dir / f"{split_name}.json", encoding="utf-8") as f:
		return json.load(f)


def process_split(entries: list[dict], images_dir: Path, out_dir: Path, split_name: str) -> list[dict]:
	out_dir.mkdir(parents=True, exist_ok=True)
	samples = []
	skipped = 0

	for entry in entries:
		text = entry["text"]
		if text is None or text.strip() == "":
			skipped += 1
			continue

		src_path = images_dir / entry["img"]
		if not src_path.exists():
			print(f"  [WARN] falta imagen {src_path}, saltando linea")
			skipped += 1
			continue

		out_path = out_dir / entry["img"]
		if not out_path.exists():
			Image.open(src_path).convert("L").save(out_path)

		samples.append({"image": str(out_path), "transcription": text})

	print(f"  [{split_name}] {len(samples)} lineas usables, {skipped} saltadas")
	return samples


def main(raw_dir: str, output_dir: str):
	raw = Path(raw_dir)
	out = Path(output_dir)

	split_dir = raw / "lines/split/basic"
	images_dir = raw / "lines/img"

	parquet_dir = out / "parquet"
	parquet_dir.mkdir(parents=True, exist_ok=True)

	for split_name in ["train", "val", "test"]:
		entries = load_split(split_dir, split_name)
		print(f"[{split_name}] {len(entries)} entradas en el JSON oficial")

		# 1/19830 train lines (0.005%) are infeasible at wr8 -- confirmed with
		# check_ctc_feasibility.py after generating the parquet. It is excluded here, not
		# rescaled (same criterion as for IAM and READ2016).
		if split_name == "train":
			entries = [e for e in entries if e["img"] != "038_1485_18.jpg"]

		samples = process_split(entries, images_dir, out / "line_images" / split_name, split_name)
		ds = Dataset.from_list(samples)
		ds = ds.cast_column("image", HFImage())
		ds.to_parquet(str(parquet_dir / f"{split_name}.parquet"))

	print(f"Done. Parquet files in {parquet_dir}")


if __name__ == "__main__":
	main(
			sys.argv[1] if len(sys.argv) > 1 else "datasets/LAM/raw/LAM",
			sys.argv[2] if len(sys.argv) > 2 else "datasets/LAM",
			)
