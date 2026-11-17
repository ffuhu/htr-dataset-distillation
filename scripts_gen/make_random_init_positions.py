"""
E5 (X2): positions in the distillation pool of the lines of the random baseline
(`data/{DS}/random_baseline_K{K}_indices.yaml`, RAW parquet INDICES as used by
`allow_indices`), to initialise DM/DC with exactly those lines (`init_positions_file`).

The distillation pool may be filtered (READ2016: `forbid_indices` of the infeasible
lines), so position != raw index in general -- the confusion behind the READ2016 medoid
index bug. HuggingFace filtering preserves order: raw index -> position is mapped and
VERIFIED line by line (image and label) against the unfiltered dataset. Fails if any line
of the baseline is not in the pool.

Usage (from the repository root):
	python scripts_gen/make_random_init_positions.py --config-name distill_read2016_dm_ctc \
		--dataset READ2016 --K 418 --out <path.yaml>
"""
import argparse
import os
import sys

import torch
import yaml
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, "src")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config-name", required=True)
	parser.add_argument("--dataset", required=True, help="Directorio en data/: IAM, GW, READ2016, LAM")
	parser.add_argument("--K", type=int, required=True)
	parser.add_argument("--out", required=True)
	parser.add_argument("--indices", help="fichero de índices crudos (por defecto data/{dataset}/random_baseline_K{K}_indices.yaml, X2)")
	args = parser.parse_args()
	assert not os.path.exists(args.out), f"{args.out} ya existe -- no se sobrescribe."

	indices_file = args.indices or f"data/{args.dataset}/random_baseline_K{args.K}_indices.yaml"
	raw_indices = [int(i) for i in yaml.safe_load(open(indices_file))]
	assert len(raw_indices) == args.K and len(set(raw_indices)) == args.K, (len(raw_indices), args.K)

	OmegaConf.register_new_resolver("int_div", lambda x, y: x // y)
	OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
	OmegaConf.register_new_resolver("eval", lambda x: eval(x))
	OmegaConf.register_new_resolver("load", OmegaConf.load)
	with initialize(version_base=None, config_path="../configs"):
		cfg = compose(config_name=args.config_name)
		filter_func = cfg.real_dataset.get("filter_func")
		if filter_func is None:
			cfg_raw = cfg
		else:
			assert filter_func._target_ == "data.huggingface.forbid_indices", \
				f"Filtro no soportado: {filter_func._target_}"
			cfg_raw = compose(config_name=args.config_name, overrides=["~real_dataset.filter_func"])
	pool = instantiate(cfg.real_dataset)
	pool_raw = instantiate(cfg_raw.real_dataset)

	if filter_func is None:
		kept = list(range(len(pool_raw)))
	else:
		forbidden = {int(i) for i in filter_func.forbidden_indices}
		kept = [r for r in range(len(pool_raw)) if r not in forbidden]
	assert len(kept) == len(pool), (len(kept), len(pool))
	raw2pos = {r: p for p, r in enumerate(kept)}

	missing = [r for r in raw_indices if r not in raw2pos]
	assert not missing, f"Líneas del baseline fuera del pool de destilación (infactibles): {missing}"
	positions = [raw2pos[r] for r in raw_indices]
	for p, r in zip(positions, raw_indices):
		a, b = pool[p], pool_raw[r]
		assert torch.equal(a["img"], b["img"]) and torch.equal(a["label"], b["label"]), \
			f"posición {p} -> índice {r} NO es la misma línea"

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	yaml.dump(positions, open(args.out, "w"))
	changed = sum(p != r for p, r in zip(positions, raw_indices))
	print(f"{args.dataset} K={args.K}: {args.out} ({changed}/{args.K} posiciones != índice crudo; "
			f"verificadas {args.K}/{args.K})")


if __name__ == "__main__":
	main()
