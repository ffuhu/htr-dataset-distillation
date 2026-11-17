import os
import platform

if platform.system() == "Darwin":
	os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import math
import random

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import affine_grid, grid_sample
from torch.utils.data import DataLoader

from omegaconf import DictConfig, OmegaConf
from hydra import main
from hydra.utils import instantiate

from lightning.pytorch import seed_everything

from data.TranscriptionCTC import to_tensordict
from code_version import stamp
from utils import get_device

def load_config(filename: str):
	return OmegaConf.load(filename)

def pool_features(features: Tensor, pooling: str, valid_widths: Tensor | None = None) -> Tensor:
	"""
	features: (batch, channels', height', width'), width' variable per batch.
	Flattens channels'xheight' into a single feature axis (same convention CRNN_CTC.forward
	uses before its RNN) and pools over the time/width axis to get a fixed-size vector.

	'multiscale': concatenated segment means at scales
	1/2/4/8 over each sample's VALID frames only (padding columns excluded via valid_widths,
	the per-sample width // width_reduction -- same criterion as CTC's input_lengths). The
	global mean makes two lines with the same characters in a different order nearly
	indistinguishable; segment means keep coarse temporal order without explicit alignment.
	"""
	features = features.transpose(1, -1).flatten(-2) # (batch, width', height'*channels')

	if pooling == "mean":
		return features.mean(dim=1)
	if pooling == "max":
		return features.max(dim=1).values
	if pooling == "multiscale":
		assert valid_widths is not None, "multiscale pooling needs per-sample valid widths"
		scales = (1, 2, 4, 8)
		pooled = []
		for i in range(features.size(0)):
			w = int(valid_widths[i].clamp(min=1, max=features.size(1)))
			# segment means via cumsum (adaptive_avg_pool1d bin bounds, which guarantee
			# non-empty segments even when w < scale; MPS lacks the non-divisible kernel)
			cum = features[i, :w].cumsum(dim=0)
			cum = torch.cat([cum.new_zeros(1, cum.size(1)), cum]) # (w+1, feat)
			starts = [(j * w) // scale for scale in scales for j in range(scale)]
			ends = [max(math.ceil((j + 1) * w / scale), (j * w) // scale + 1)
					for scale in scales for j in range(scale)]
			starts = torch.tensor(starts, device=features.device)
			ends = torch.tensor(ends, device=features.device)
			segments = (cum[ends] - cum[starts]) / (ends - starts).unsqueeze(1) # (15, feat)
			pooled.append(segments.flatten())
		return torch.stack(pooled)

	raise ValueError(f"Unknown pooling mode: '{pooling}' (expected 'mean', 'max' or 'multiscale').")

def dsa_affine(imgs: Tensor, angle_degrees: float, tx: float, ty: float) -> Tensor:
	"""
	DSA-CTC: the SAME sampled RandomAffine parameters
	applied to the real and synthetic branches of one matching iteration, so the synthetic
	pixels must encode augmentation-invariant content instead of brittle shortcuts.
	affine_grid + grid_sample are differentiable w.r.t. the input, so the gradient path to
	the synthetic pixels is preserved (unlike torchvision's RandomAffine).

	angle_degrees rotates about the image center; tx/ty are translation FRACTIONS of
	width/height (torchvision RandomAffine convention). The matrix maps output to input
	coordinates in affine_grid's normalized space, with the aspect-ratio correction (H/W,
	W/H) -- without it, a 3-degree rotation on a ~15:1 text line becomes a huge shear.
	Padding is zeros, matching torchvision's default fill.
	"""
	_, _, H, W = imgs.shape
	rad = math.radians(angle_degrees)
	cos, sin = math.cos(rad), math.sin(rad)
	matrix = imgs.new_tensor([
			[cos, sin * H / W, -2.0 * tx],
			[-sin * W / H, cos, -2.0 * ty],
			]).unsqueeze(0).expand(imgs.size(0), -1, -1)
	grid = affine_grid(matrix, list(imgs.shape), align_corners=False)
	return grid_sample(imgs, grid, align_corners=False, padding_mode="zeros")

def freeze(module: Module) -> Module:
	for p in module.parameters():
		p.requires_grad_(False)
	return module

@main(version_base=None, config_path="../configs", config_name="distill_iam_dm_ctc.yaml")
def _main(cfg: DictConfig):
	stamp()  # code commit; fails if it does not match CODE_VERSION
	if "seed" in cfg:
		seed_everything(cfg.seed)
	rng = random.Random(cfg.seed)

	# This script is plain PyTorch (no lightning.pytorch.Trainer, which auto-detects the
	# accelerator) -- without this, every tensor defaults to CPU regardless of what's available.
	device = get_device()
	print(f"Device: {device}")

	real_dataset = instantiate(cfg.real_dataset)
	print(f"Real (filtered) training subset: {len(real_dataset)} samples")

	real_loader = DataLoader(
			real_dataset,
			batch_size=cfg.matching_batch_size,
			shuffle=True,
			collate_fn=real_dataset.collate_fn,
			)
	real_iter = iter(real_loader)

	def next_real_batch():
		nonlocal real_iter
		try:
			return next(real_iter)
		except StopIteration:
			real_iter = iter(real_loader)
			return next(real_iter)

	# --- Initialize the synthetic buffer: same convention as DC-CTC (fixed inherited labels,
	# fixed per-sample width, only pixels are learnable). ---
	#
	# Conditional matching (pseudo-classes): with
	# cluster_file set, each synthetic sample is anchored to one text cluster (init = the
	# cluster medoid) and its matching target is the mean embedding of *that cluster only*
	# -- the analogue of DM's per-class mean matching. Without it, the unconditional mean
	# embedding over heterogeneous sequences is a contentless centroid.
	#
	# conditional_matching (ablation): decoupled from
	# cluster_file on purpose. cluster_file alone controls WHERE the buffer initializes
	# from (medoids); conditional_matching controls whether the outer loop matches
	# per-cluster (True) or globally (False). Setting cluster_file + conditional_matching=
	# false isolates "does the conditioned matching loop add anything beyond a good
	# (medoid) initialization?" -- a confound that unconditional buffers with random
	# (non-medoid) init cannot separate.
	# Defaults to the historical behavior: conditional iff a cluster_file is given.
	conditional_matching = bool(cfg.get("conditional_matching", cfg.get("cluster_file") is not None))
	# round_robin_syn (ablation): with conditional_matching=false the historical
	# global loop updates min(matching_batch_size, K) random synthetics per iteration, i.e.
	# min(matching_batch_size, K)x more pixel updates per synthetic than the conditional loop
	# (1 synthetic per iteration, round-robin) -- a dose confound (see the drift
	# dose-response). round_robin_syn=true keeps the conditional loop's update schedule (synthetic
	# cid = outer_it % K, alone) and changes ONLY where the real batch comes from: a random
	# batch from the global pool instead of the synthetic's own cluster. Default False =
	# historical behavior.
	round_robin_syn = bool(cfg.get("round_robin_syn", False))
	assert not (round_robin_syn and conditional_matching), \
		"round_robin_syn=true only applies to the unconditional loop (conditional_matching=false)."
	cluster_members: list[list[int]] | None = None
	if cfg.get("cluster_file"):
		cluster_cfg = load_config(cfg.cluster_file)
		assert int(cluster_cfg.k) == int(cfg.K), \
			f"cluster_file has k={cluster_cfg.k} but K={cfg.K} -- one synthetic sample per cluster."
		assignments = [int(a) for a in cluster_cfg.clusters]
		assert len(assignments) == len(real_dataset), (len(assignments), len(real_dataset))
		source_indices = [int(m) for m in cluster_cfg.medoids]
		if conditional_matching:
			cluster_members = [[p for p, a in enumerate(assignments) if a == c] for c in range(cfg.K)]
			sizes = [len(m) for m in cluster_members]
			print(f"Conditional matching: {cfg.K} clusters from '{cfg.cluster_file}' "
					f"(sizes min={min(sizes)} max={max(sizes)})")
		else:
			print(f"Medoid-initialized, UNCONDITIONAL matching: {cfg.K} synthetics initialized "
					f"from '{cfg.cluster_file}' medoids; matching target = global pool batches "
					f"(conditioning ablation).")
	else:
		assert not conditional_matching, \
			"conditional_matching=true requires cluster_file (need cluster assignments to match by)."
		if cfg.get("init_positions_file"):
			# E5 (X2): init from a given set of pool POSITIONS (not raw parquet indices;
			# produced and verified line by line by scripts_gen/make_random_init_positions.py).
			source_indices = [int(p) for p in load_config(cfg.init_positions_file)]
			assert len(source_indices) == cfg.K and len(set(source_indices)) == cfg.K, \
				f"init_positions_file must hold K={cfg.K} distinct positions, got {len(source_indices)}"
			assert all(0 <= p < len(real_dataset) for p in source_indices)
			print(f"Initialized from {cfg.K} pool positions in '{cfg.init_positions_file}'.")
		else:
			source_indices = rng.sample(range(len(real_dataset)), cfg.K)

	synthetic_images: list[torch.nn.Parameter] = []
	synthetic_labels: list[Tensor] = []
	for idx in source_indices:
		sample = real_dataset[idx]
		synthetic_images.append(torch.nn.Parameter(sample["img"].clone()))
		synthetic_labels.append(sample["label"].clone())

	def sample_synthetic_batch(batch_size: int):
		idxs = rng.sample(range(cfg.K), min(batch_size, cfg.K))
		items = [to_tensordict(synthetic_images[i], synthetic_labels[i], "synthetic") for i in idxs]
		return real_dataset.collate_fn(items)

	syn_optimizer = torch.optim.Adam(synthetic_images, lr=cfg.syn_lr)

	# --- Embedding encoder: two modes ---
	#
	# 'random' -- untrained network, frozen (no grad on its own weights) but still
	# differentiable w.r.t. its input, and periodically re-drawn from scratch (fresh random
	# init), exactly as in the original DM (Zhao & Bilen, WACV 2023).
	#
	# 'teacher' -- a DELIBERATE deviation from original DM (towards the trained-feature
	# variants: CAFE/IDM/DataDAM): the frozen CNN of the trained wr8 teacher. Motivated by
	# the drift dose-response finding: the periodic re-draw makes the matching target
	# non-stationary -- pixels chase means of ever-changing random encoders and what gets
	# written is destructive noise. The teacher CNN is a stationary, content-bearing target.
	# eval() matters: BatchNorm must use the teacher's running stats, not batch stats.
	encoder_source = cfg.get("encoder_source", "random")
	if encoder_source == "teacher":
		from models.CRNN_CTC_Wrapper import CRNN_CTC_Wrapper
		teacher = CRNN_CTC_Wrapper.load_from_checkpoint(cfg.teacher_ckpt, map_location="cpu", weights_only=False)
		embedding_encoder = freeze(teacher.model.encoder).eval().to(device)
		print(f"Teacher encoder: '{cfg.teacher_ckpt}' "
				f"(width_reduction={embedding_encoder.width_reduction}, stationary -- no resampling)")
	elif encoder_source == "random":
		embedding_encoder = freeze(instantiate(cfg.embedding_encoder)).to(device)
	else:
		raise ValueError(f"Unknown encoder_source: '{encoder_source}' (expected 'random' or 'teacher').")

	use_dsa = bool(cfg.get("dsa", False))
	if use_dsa:
		print(f"DSA enabled: RandomAffine +-{cfg.dsa_rotation_degrees} deg / +-{cfg.dsa_translation} "
				f"translation, p={cfg.dsa_chance}, shared parameters across both branches")

	# E6 (X1): optional intermediate buffers at given outer iterations (dose sweep in a
	# single run); saved as <buffer_path stem>_it<N>.pt with the same format as the final one.
	snapshot_iterations = {int(i) for i in (cfg.get("snapshot_iterations") or [])}
	assert all(0 < i <= cfg.outer_iterations for i in snapshot_iterations), \
		f"snapshot_iterations must lie in [1, outer_iterations={cfg.outer_iterations}]"

	def save_buffer(path: str):
		buffer = {
				"images": [p.detach().clone() for p in synthetic_images],
				"labels": [label.clone() for label in synthetic_labels],
				"source_indices": source_indices,
				}
		buffer_dir = os.path.dirname(path)
		if buffer_dir:
			os.makedirs(buffer_dir, exist_ok=True)
		torch.save(buffer, path)
		print(f"Saved DM-distilled buffer ({cfg.K} samples) to '{path}'.")

	for outer_it in range(cfg.outer_iterations):
		if encoder_source == "random" and outer_it % cfg.encoder_resample_every == 0:
			embedding_encoder = freeze(instantiate(cfg.embedding_encoder)).to(device)

		if cluster_members is not None:
			cid = outer_it % cfg.K
			members = cluster_members[cid]
			idxs = rng.sample(members, min(cfg.matching_batch_size, len(members)))
			B_real = real_dataset.collate_fn([real_dataset[i] for i in idxs]).to(device)
			B_syn = real_dataset.collate_fn(
					[to_tensordict(synthetic_images[cid], synthetic_labels[cid], "synthetic")]).to(device)
		elif round_robin_syn:
			cid = outer_it % cfg.K
			B_real = next_real_batch().to(device)
			B_syn = real_dataset.collate_fn(
					[to_tensordict(synthetic_images[cid], synthetic_labels[cid], "synthetic")]).to(device)
		else:
			B_real = next_real_batch().to(device)
			B_syn = sample_synthetic_batch(cfg.matching_batch_size).to(device)

		real_img, syn_img = B_real["img"], B_syn["img"]
		if use_dsa and rng.random() < cfg.dsa_chance:
			angle = rng.uniform(-cfg.dsa_rotation_degrees, cfg.dsa_rotation_degrees)
			tx = rng.uniform(-cfg.dsa_translation, cfg.dsa_translation)
			ty = rng.uniform(-cfg.dsa_translation, cfg.dsa_translation)
			real_img = dsa_affine(real_img, angle, tx, ty)
			syn_img = dsa_affine(syn_img, angle, tx, ty)

		# valid frames per sample = width // width_reduction, from each sample's own shape
		# (only used by multiscale pooling; mean/max keep their historical behavior)
		wr = int(embedding_encoder.width_reduction)
		features_real = pool_features(embedding_encoder(real_img), cfg.pooling, B_real["shape"][:, 1] // wr)
		features_syn = pool_features(embedding_encoder(syn_img), cfg.pooling, B_syn["shape"][:, 1] // wr)

		# Mean-embedding distance: the practical simplification of MMD that DM uses with a
		# fixed/random embedding network (no CTC anywhere in this loop).
		matching_loss = (features_real.mean(dim=0) - features_syn.mean(dim=0)).pow(2).sum()

		syn_optimizer.zero_grad(set_to_none=True)
		matching_loss.backward()
		syn_optimizer.step()

		if outer_it + 1 in snapshot_iterations:
			save_buffer(f"{os.path.splitext(cfg.buffer_path)[0]}_it{outer_it + 1}.pt")

		if outer_it % cfg.log_every == 0:
			print(f"[{outer_it:4d}/{cfg.outer_iterations}] matching_loss={matching_loss.item():.6f}")

	save_buffer(cfg.buffer_path)

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
