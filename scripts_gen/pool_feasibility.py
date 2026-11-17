"""
CTC feasibility (T = width // wr >= L + repeats) in the full training pools (unfiltered) and
membership of infeasible lines in the subsets used in the paper (raw indices: random_baseline_K*,
medoids_k*, x5_random_K*). Motivation: an audit of the buffers found infeasible lines in IAM (MTT
init of seed 0; medoids K=616), which was assumed to be feasible at wr=8.

Usage (from the repository root):  python scripts_gen/pool_feasibility.py --config-name distill_iam_dm_ctc_wr8 --dataset IAM
"""
import argparse
import glob
import os
import sys

import yaml
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, "src")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config-name", required=True)
	parser.add_argument("--dataset", required=True)
	parser.add_argument("--wr", type=int, default=8)
	args = parser.parse_args()

	OmegaConf.register_new_resolver("int_div", lambda x, y: x // y)
	OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
	OmegaConf.register_new_resolver("eval", lambda x: eval(x))
	OmegaConf.register_new_resolver("load", OmegaConf.load)
	with initialize(version_base=None, config_path="../configs"):
		cfg = compose(config_name=args.config_name)
		if cfg.real_dataset.get("filter_func") is not None:
			cfg = compose(config_name=args.config_name, overrides=["~real_dataset.filter_func"])
	pool = instantiate(cfg.real_dataset)

	bad = []
	for i in range(len(pool)):
		s = pool[i]
		T = s["img"].shape[-1] // args.wr
		lab = s["label"]
		L = int(lab.shape[0])
		rep = int((lab[1:] == lab[:-1]).sum()) if L > 1 else 0
		if L == 0 or T < L + rep:
			bad.append(i)
	print(f"{args.dataset}: pool crudo {len(pool)} líneas; infactibles a wr={args.wr}: {len(bad)} -> {bad}")
	out = f"data/{args.dataset}/infeasible_wr{args.wr}_indices_audit.yaml"
	yaml.dump(bad, open(out, "w"))

	bad_set = set(bad)
	files = sorted(glob.glob(f"data/{args.dataset}/random_baseline_K*_indices.yaml")
				+ glob.glob(f"data/{args.dataset}/medoids_k*_indices.yaml")
				+ glob.glob(f"data/{args.dataset}/x5_random_K*_indices.yaml"))
	for f in files:
		idx = [int(i) for i in yaml.safe_load(open(f))]
		hit = sorted(bad_set & set(idx))
		print(f"  {os.path.basename(f)}: K={len(idx)} infactibles={len(hit)} {hit if hit else ''}")


if __name__ == "__main__":
	main()
