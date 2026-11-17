"""
Pixel drift of the distilled buffers of the budget sweep with respect to their initialisation,
measured against the 0-iteration buffer (`outer_iterations=0`: the medoids as they are, saved
by the same distillation script).

Relative L2 per image, ||buf - init|| / ||init||, comparing buffer against buffer: no need to
instantiate the dataset, and the initialisation is exactly the one the distillation saw.

Usage:
	python src/buffer_drift_budget.py --init <zeroiter.pt> --buffers <a.pt> <b.pt> ...
"""
import argparse

import torch


def load(path):
	return torch.load(path, map_location="cpu", weights_only=False)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--init", required=True)
	parser.add_argument("--buffers", nargs="+", required=True)
	args = parser.parse_args()

	init = load(args.init)
	for path in args.buffers:
		buf = load(path)
		assert list(buf["source_indices"]) == list(init["source_indices"]), \
			f"{path}: source_indices distintos de {args.init}"
		assert len(buf["images"]) == len(init["images"]), (len(buf["images"]), len(init["images"]))
		rels = []
		for b, i in zip(buf["images"], init["images"]):
			assert b.shape == i.shape, f"{path}: shape {tuple(b.shape)} != init {tuple(i.shape)}"
			rels.append(((b.float() - i.float()).norm() / i.float().norm()).item())
		r = torch.tensor(rels)
		print(f"DRIFT {path} n={len(rels)} media={r.mean():.5f} mediana={r.median():.5f} "
				f"min={r.min():.5f} max={r.max():.5f}")


if __name__ == "__main__":
	main()
