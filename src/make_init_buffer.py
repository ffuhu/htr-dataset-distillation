"""
"0-iteration" buffer for an existing distilled buffer whose initialisation was not saved
separately (MTT of the sweep: random init per seed). Same lines (`source_indices`) and same
labels, but with the real pixels of the pool (instantiated with the SAME distillation
config: same filtering and order). It is the MTT analogue of the `*_dm_zeroiter_*` buffer of
DM/DC.

Usage:
	python src/make_init_buffer.py --config-name distill_gw_mtt_ctc \
		--pairs in_s0.pt:out_s0.pt in_s42.pt:out_s42.pt ...
"""
import argparse
import os

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config-name", required=True)
	parser.add_argument("--pairs", nargs="+", required=True, help="entrada.pt:salida.pt")
	args = parser.parse_args()

	OmegaConf.register_new_resolver("int_div", lambda x, y: x // y)
	OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
	OmegaConf.register_new_resolver("eval", lambda x: eval(x))
	OmegaConf.register_new_resolver("load", OmegaConf.load)
	with initialize(version_base=None, config_path="../configs"):
		cfg = compose(config_name=args.config_name)
	pool = instantiate(cfg.real_dataset)
	print(f"Pool '{args.config_name}': {len(pool)} líneas")

	for pair in args.pairs:
		src_path, dst_path = pair.split(":")
		assert not os.path.exists(dst_path), f"{dst_path} ya existe -- no se sobrescribe."
		buf = torch.load(src_path, map_location="cpu", weights_only=False)
		images, labels = [], []
		for img, label, src in zip(buf["images"], buf["labels"], buf["source_indices"]):
			sample = pool[int(src)]
			assert sample["img"].shape == img.shape, f"{src_path}[{src}]: shape distinta"
			assert torch.equal(sample["label"], label), f"{src_path}[{src}]: etiqueta distinta"
			images.append(sample["img"].clone())
			labels.append(label.clone())
		torch.save({"images": images, "labels": labels, "source_indices": list(buf["source_indices"])}, dst_path)
		print(f"{src_path} -> {dst_path} ({len(images)} líneas)")


if __name__ == "__main__":
	main()
