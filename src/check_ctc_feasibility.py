import sys

from hydra import main
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

def load_config(filename: str):
	return OmegaConf.load(filename)

@main(version_base=None, config_path="../configs", config_name="distill_iam_dc_ctc.yaml")
def _main(cfg: DictConfig):
	"""
	Pre-flight check: every line in the real pool
	must satisfy the CTC feasibility condition T >= L + n_consecutive_repeats, where
	T = floor(width / width_reduction). Each pair of equal consecutive label tokens needs an
	extra blank frame between them, hence the repeats term -- plain T >= L is not enough.

	Run with the same config and overrides as the distillation run it gates, e.g.:
		python src/check_ctc_feasibility.py --config-name distill_read2016_dc_ctc

	A violated sample would not crash anything downstream: with zero_infinity=True its CTC
	loss becomes 0 and it silently stops contributing gradient (both in the synthetic branch
	of distill_dc_ctc.py and in the final evaluation training) -- which is exactly the kind of
	contamination this check exists to rule out before spending compute.
	"""
	width_reduction = int(cfg.inner_model.modules.encoder.width_reduction)

	real_dataset = instantiate(cfg.real_dataset)
	print(f"Pool: {len(real_dataset)} samples; width_reduction={width_reduction}")

	violations: list[tuple[int, int, int, int]] = []
	min_margin: int | None = None
	for idx in range(len(real_dataset)):
		sample = real_dataset[idx]
		label = sample["label"]

		T = sample["img"].shape[2] // width_reduction
		L = label.shape[0]
		repeats = int((label[1:] == label[:-1]).sum()) if L > 1 else 0

		margin = T - (L + repeats)
		if min_margin is None or margin < min_margin:
			min_margin = margin
		if margin < 0:
			violations.append((idx, T, L, repeats))

	print(f"Min margin (T - (L + consecutive repeats)) across pool: {min_margin}")
	if violations:
		print(f"[FAIL] {len(violations)} sample(s) infeasible for CTC at width_reduction={width_reduction}:")
		for idx, T, L, repeats in violations:
			print(f"  pool idx={idx}: T={T} < L + repeats = {L} + {repeats} = {L + repeats}")
		print("Decide how to treat these lines (exclude from the pool / rescale) BEFORE distilling.")
		sys.exit(1)
	print("[OK] every sample satisfies T >= L + consecutive repeats.")

if __name__ == "__main__":
	OmegaConf.register_new_resolver("int_div", lambda x, y: x//y)
	OmegaConf.register_new_resolver("mul", lambda x, y: x*y)
	OmegaConf.register_new_resolver("eval", lambda x: eval(x))
	OmegaConf.register_new_resolver("load", load_config)
	_main()
	OmegaConf.clear_resolver("int_div")
	OmegaConf.clear_resolver("mul")
	OmegaConf.clear_resolver("eval")
	OmegaConf.clear_resolver("load")
