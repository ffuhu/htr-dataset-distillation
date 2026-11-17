"""
Fix of the READ2016 medoid indices.

`gen_indices.sh` dumps `text_clusters_k{K}.yaml:medoids` (POSITIONS in the pool filtered of the
infeasible lines in `infeasible_wr8_indices.yaml`) to `medoids_k{K}_indices.yaml`, which
`allow_indices` reads as RAW parquet INDICES. This script translates position -> raw index
(HuggingFace filtering preserves order) and VERIFIES it line by line:
image and label of filtered_pool[position] == unfiltered_dataset[raw index].
The previous file is kept as `.buggy_bak` (not deleted). The indices in data/READ2016/ are
already fixed; run this only right after `gen_indices.sh`.

Usage (from the repository root):
	python scripts_gen/fix_read2016_medoid_indices.py
"""
import os
import sys

import torch
import yaml
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, "src")

KS = [84, 418, 835]


def main():
	forbidden = sorted(int(i) for i in yaml.safe_load(open("data/READ2016/infeasible_wr8_indices.yaml")))

	def pos2raw(p: int) -> int:
		r = p
		for f in forbidden:
			if f <= r:
				r += 1
			else:
				break
		return r

	OmegaConf.register_new_resolver("int_div", lambda x, y: x // y)
	OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
	OmegaConf.register_new_resolver("eval", lambda x: eval(x))
	OmegaConf.register_new_resolver("load", OmegaConf.load)
	with initialize(version_base=None, config_path="../configs"):
		cfg_f = compose(config_name="distill_read2016_dm_ctc")
		cfg_raw = compose(config_name="distill_read2016_dm_ctc", overrides=["~real_dataset.filter_func"])
	pool_f = instantiate(cfg_f.real_dataset)
	pool_raw = instantiate(cfg_raw.real_dataset)
	print(f"Pool filtrado: {len(pool_f)}; sin filtrar: {len(pool_raw)}; infactibles: {len(forbidden)}")
	assert len(pool_raw) - len(pool_f) == len(forbidden)

	for K in KS:
		out = f"data/READ2016/medoids_k{K}_indices.yaml"
		bak = out + ".buggy_bak"
		assert not os.path.exists(bak), f"{bak} ya existe -- ¿ya se aplicó el arreglo?"
		positions = [int(p) for p in yaml.safe_load(open(f"data/READ2016/text_clusters_k{K}.yaml"))["medoids"]]
		old = [int(i) for i in yaml.safe_load(open(out))]
		assert old == positions, f"{out} no coincide con las posiciones de text_clusters -- estado inesperado."
		raw = [pos2raw(p) for p in positions]
		assert len(set(raw)) == K and not (set(raw) & set(forbidden))
		for p, r in zip(positions, raw):
			a, b = pool_f[p], pool_raw[r]
			assert torch.equal(a["img"], b["img"]) and torch.equal(a["label"], b["label"]), \
				f"K={K}: posición {p} -> índice {r} NO es la misma línea"
		os.rename(out, bak)
		yaml.dump(raw, open(out, "w"))
		changed = len(set(raw) - set(old))
		print(f"K={K}: {out} corregido ({changed}/{K} líneas cambian; verificadas {K}/{K}); anterior en {bak}")


if __name__ == "__main__":
	main()
