"""
Pixel drift of buffers whose initialisation was not saved separately (MTT of the sweep:
random init per seed). Compares each image of the buffer with the real image of its
`source_index` in the pool, instantiated with the SAME distillation config (same filtering
and order). Same metric as src/buffer_drift_budget.py (relative L2 per image).

Usage:
	python src/buffer_drift_pool.py --config-name distill_read2016_mtt_ctc --buffers a.pt b.pt
"""
import argparse

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config-name", required=True)
	parser.add_argument("--buffers", nargs="+", required=True)
	args = parser.parse_args()

	OmegaConf.register_new_resolver("int_div", lambda x, y: x // y)
	OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
	OmegaConf.register_new_resolver("eval", lambda x: eval(x))
	OmegaConf.register_new_resolver("load", OmegaConf.load)
	with initialize(version_base=None, config_path="../configs"):
		cfg = compose(config_name=args.config_name)
	pool = instantiate(cfg.real_dataset)
	print(f"Pool '{args.config_name}': {len(pool)} líneas")

	for path in args.buffers:
		buf = torch.load(path, map_location="cpu", weights_only=False)
		assert len(buf["images"]) == len(buf["source_indices"])
		rels = []
		for img, src in zip(buf["images"], buf["source_indices"]):
			init = pool[int(src)]["img"]
			assert init.shape == img.shape, f"{path}: shape {tuple(img.shape)} != init {tuple(init.shape)}"
			rels.append(((img.float() - init.float()).norm() / init.float().norm()).item())
		r = torch.tensor(rels)
		print(f"DRIFT {path} n={len(rels)} media={r.mean():.5f} mediana={r.median():.5f} "
				f"min={r.min():.5f} max={r.max():.5f}")


if __name__ == "__main__":
	main()
