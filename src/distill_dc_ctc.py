import os
import platform

# CTC loss is not available on MPS in current PyTorch versions; allow CPU fallback.
if platform.system() == "Darwin":
	os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import random
from functools import partial

import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader

from omegaconf import DictConfig, OmegaConf
from hydra import main
from hydra.utils import instantiate

from lightning.pytorch import seed_everything

from data.TranscriptionCTC import to_tensordict
from code_version import stamp
from utils import differentiable_ctc_loss, get_device

def load_config(filename: str):
	return OmegaConf.load(filename)

def ctc_loss_for_batch(batch, model, loss_function, device) -> Tensor:
	batch = batch.to(device)
	X = batch["img"]
	Y = batch["label"]
	W = batch["shape"][:, 1]
	L = batch["length"]

	logits, *_ = model(X)
	logits = logits.permute(1, 0, 2).log_softmax(dim=-1)
	input_lengths = W.div(model.encoder.width_reduction, rounding_mode="trunc")

	return loss_function(logits, Y, input_lengths, L)

def dc_original_matching_loss(grads_real: list[Tensor], grads_syn: list[Tensor]) -> Tensor:
	"""
	Distance of the original DC code (VICO-UoE/DatasetCondensation, utils.py `distance_wb`,
	dis_metric='ours'): per output row, 1 - <gr, gs> / (||gr|| ||gs|| + 1e-6); 1-D tensors
	(biases, BatchNorm affine) contribute 0. The additive 1e-6 bounds the derivative w.r.t.
	near-zero grad_syn rows, which in batch mode (round_robin_syn=false) reach ~3e-8 in
	LSTM.weight_hh_l0_reverse and overflow the double backward to NaN with the default
	distance below.
	"""
	total = grads_real[0].new_zeros(())
	for g_real, g_syn in zip(grads_real, grads_syn):
		if g_real.dim() == 1:
			continue
		g_real_flat = g_real.reshape(g_real.shape[0], -1)
		g_syn_flat = g_syn.reshape(g_syn.shape[0], -1)
		cos = (g_real_flat * g_syn_flat).sum(dim=1) / (g_real_flat.norm(dim=1) * g_syn_flat.norm(dim=1) + 1e-6)
		total = total + (1.0 - cos).sum()
	return total

def gradient_matching_loss(grads_real: list[Tensor], grads_syn: list[Tensor]) -> Tensor:
	"""
	Per-parameter-tensor row-wise cosine distance (DC, Zhao et al. 2021), summed over every
	parameter tensor of the model. Each row of a >=2D tensor is treated as the gradient
	reaching one output unit (conv output channel, LSTM gate row, ...); 1D tensors (biases,
	BatchNorm affine params) are treated as a single row.
	"""
	total: Tensor
	total_started = False

	for g_real, g_syn in zip(grads_real, grads_syn):
		g_real_flat = g_real.reshape(g_real.shape[0], -1) if g_real.dim() >= 2 else g_real.reshape(1, -1)
		g_syn_flat = g_syn.reshape(g_syn.shape[0], -1) if g_syn.dim() >= 2 else g_syn.reshape(1, -1)

		layer_distance = (1.0 - cosine_similarity(g_real_flat, g_syn_flat, dim=1, eps=1e-8)).sum()
		total = layer_distance if not total_started else total + layer_distance
		total_started = True

	assert total_started, "Model has no parameters to match gradients against."
	return total

@main(version_base=None, config_path="../configs", config_name="distill_iam_dc_ctc.yaml")
def _main(cfg: DictConfig):
	stamp()  # code commit; fails if it does not match CODE_VERSION
	if "seed" in cfg:
		seed_everything(cfg.seed)
	rng = random.Random(cfg.seed)

	# This script is plain PyTorch (no lightning.pytorch.Trainer, which auto-detects the
	# accelerator) -- without this, every tensor defaults to CPU regardless of what's available.
	# require_double_backward=True: MPS's max_pool2d has no second derivative, which the
	# synthetic branch below needs (create_graph=True) -- see utils.device.get_device.
	device = get_device(require_double_backward=True)
	print(f"Device: {device}")

	real_dataset = instantiate(cfg.real_dataset)
	print(f"Real (filtered) training subset: {len(real_dataset)} samples")

	cfg.max_h = real_dataset.stats["max_h"][0]
	cfg.max_w = real_dataset.stats["max_w"][0]
	cfg.max_l = real_dataset.stats["max_l"][0]
	cfg.t2i = real_dataset.class_to_int
	cfg.i2t = real_dataset.int_to_class
	cfg.vocabulary_size = len(cfg.i2t)

	loss_function = instantiate(cfg.loss_function)

	def build_inner_model_and_optimizer():
		model = instantiate(cfg.inner_model).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=cfg.inner_lr)
		return model, optimizer

	inner_model, inner_optimizer = build_inner_model_and_optimizer()
	params = list(inner_model.parameters())

	# The synthetic branch needs create_graph=True through its own gradient (grad_syn), which
	# torch.nn.CTCLoss cannot do (no double-backward: "derivative for aten::_ctc_loss_backward
	# is not implemented"). differentiable_ctc_loss
	# is a from-scratch, finite-difference-validated CTC (utils/differentiable_ctc.py) built
	# only from ops that do support a second derivative. It is much slower than the built-in
	# CTCLoss, so it is used *only* here -- the real branch and the co-adaptation step below
	# don't need a second derivative and keep using the fast `loss_function`.
	synthetic_loss_function = partial(
			differentiable_ctc_loss,
			blank=cfg.t2i["<CTC>"],
			reduction="mean",
			zero_infinity=True,
			)

	print(inner_model)
	print(f"Inner (distillation) model parameters: {sum(p.numel() for p in params):,}")

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

	# --- Initialize the synthetic buffer: K pixels-as-parameters, fixed inherited labels ---
	# Width is never learned (fixed to the source real line's width), and labels are frozen
	# (never wrapped in a Parameter, never part of any optimizer).
	#
	# Conditional matching (pseudo-classes): with
	# cluster_file set, each synthetic sample is anchored to one text cluster (init = the
	# cluster medoid, labels inherited from it) and its matching target is the mean gradient
	# over lines of *that cluster only* -- the analogue of DC's per-class matching. Without
	# it, the unconditional mean over heterogeneous sequences cancels the content-specific
	# gradient components and the pixels only receive generic statistics.
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

	def sample_synthetic_batch(batch_size: int, detach: bool = False):
		idxs = rng.sample(range(cfg.K), min(batch_size, cfg.K))
		items = [
				to_tensordict(
						synthetic_images[i].detach() if detach else synthetic_images[i],
						synthetic_labels[i],
						"synthetic",
						)
				for i in idxs
				]
		return real_dataset.collate_fn(items)

	syn_optimizer = torch.optim.Adam(synthetic_images, lr=cfg.syn_lr)

	# Gradient-matching distance. "cosine" (default) = historical behavior of every
	# DC run so far; "dc_original" = distance_wb of the original DC code (see
	# dc_original_matching_loss), numerically stable in batch mode.
	match_distance = cfg.get("match_distance", "cosine")
	assert match_distance in ("cosine", "dc_original"), f"Unknown match_distance: {match_distance}"
	match_distance_fn = gradient_matching_loss if match_distance == "cosine" else dc_original_matching_loss
	print(f"Gradient-matching distance: {match_distance}")
	skip_nonfinite_syn_grad = bool(cfg.get("skip_nonfinite_syn_grad", False))
	n_skipped_syn = 0

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
		n_nonfinite = sum(int((~torch.isfinite(img)).sum()) for img in buffer["images"])
		print(f"Saved distilled buffer ({cfg.K} samples) to '{path}'. Non-finite pixels: {n_nonfinite}")

	for outer_it in range(cfg.outer_iterations):
		# Periodic full reinit of theta to a fresh random point (Algorithm 1 of DC, Zhao et
		# al. 2021: several restarts from random theta_0, each followed by a few co-adaptation
		# steps). Without this, the single-trajectory co-adaptation loop below can drive theta
		# into a shared low-gradient plateau for both the real and synthetic batches -- observed
		# empirically in a K=20/500-line diagnostic: matching_loss climbed past 1000 while
		# ||grad_syn|| collapsed towards zero from iteration ~30 onwards, once theta stopped moving.
		# Reinitializing periodically keeps theta -- and therefore the matching gradient -- in a
		# regime informative enough to keep updating the pixels for the whole run, and also
		# prevents the synthetic buffer from overfitting to a single theta trajectory.
		if outer_it > 0 and outer_it % cfg.reinit_every_n == 0:
			inner_model, inner_optimizer = build_inner_model_and_optimizer()
			params = list(inner_model.parameters())
			print(f"[{outer_it:4d}/{cfg.outer_iterations}] reinitialized theta (fresh random init)")

		if cluster_members is not None:
			# One cluster per outer iteration, round-robin (DC iterates classes the same way):
			# the target is the cluster's mean gradient, the synthetic branch is that
			# cluster's single synthetic sample.
			cid = outer_it % cfg.K
			members = cluster_members[cid]
			idxs = rng.sample(members, min(cfg.matching_batch_size, len(members)))
			B_real = real_dataset.collate_fn([real_dataset[i] for i in idxs])
			B_syn = real_dataset.collate_fn(
					[to_tensordict(synthetic_images[cid], synthetic_labels[cid], "synthetic")])
		elif round_robin_syn:
			cid = outer_it % cfg.K
			B_real = next_real_batch()
			B_syn = real_dataset.collate_fn(
					[to_tensordict(synthetic_images[cid], synthetic_labels[cid], "synthetic")])
		else:
			B_real = next_real_batch()
			B_syn = sample_synthetic_batch(cfg.matching_batch_size)

		# Real branch: target gradient direction. No need to reach past theta, so no
		# create_graph here -- this is the "target" the synthetic branch is pulled towards.
		loss_real = ctc_loss_for_batch(B_real, inner_model, loss_function, device)
		grad_real = torch.autograd.grad(loss_real, params, create_graph=False)

		# Synthetic branch: create_graph=True keeps grad_syn connected to the synthetic
		# pixels through the forward pass, so matching_loss.backward() below can reach them.
		# cuDNN's LSTM backward has no second derivative ("_cudnn_rnn_backward" double-backward
		# not implemented), so cuDNN must be disabled for this forward pass only -- it forces
		# PyTorch's native (non-cuDNN) RNN kernel, which does support create_graph=True. The real
		# branch and co-adaptation step below don't need a second derivative and keep using cuDNN.
		with torch.backends.cudnn.flags(enabled=False):
			loss_syn = ctc_loss_for_batch(B_syn, inner_model, synthetic_loss_function, device)
		grad_syn = torch.autograd.grad(loss_syn, params, create_graph=True)

		matching_loss = match_distance_fn(grad_real, grad_syn)

		syn_optimizer.zero_grad(set_to_none=True)
		matching_loss.backward()
		if skip_nonfinite_syn_grad and not all(
				torch.isfinite(p.grad).all() for p in synthetic_images if p.grad is not None):
			# E6 (X1b): the second-order CTC gradient occasionally overflows in batch mode;
			# skip that pixel update instead of writing NaN into the buffer. The
			# rest of the iteration (snapshots, co-adaptation) runs unchanged; the count is reported.
			n_skipped_syn += 1
			print(f"[WARN] [{outer_it:4d}/{cfg.outer_iterations}] non-finite synthetic pixel gradient "
					f"-- pixel update skipped ({n_skipped_syn} so far).")
		else:
			syn_optimizer.step()

		if outer_it + 1 in snapshot_iterations:
			save_buffer(f"{os.path.splitext(cfg.buffer_path)[0]}_it{outer_it + 1}.pt")

		if outer_it % cfg.log_every == 0:
			# ||grad_syn|| is logged separately from matching_loss: if theta memorizes the
			# tiny synthetic buffer during co-adaptation, grad_syn collapses towards zero and
			# matching_loss stalls with no signal reaching the pixels, without erroring out.
			# matching_loss stalling *together with* a collapsing ||grad_syn|| is the
			# signature of that failure mode.
			grad_syn_norm = torch.cat([g.detach().reshape(-1) for g in grad_syn]).norm().item()
			print(
					f"[{outer_it:4d}/{cfg.outer_iterations}] "
					f"matching_loss={matching_loss.item():.4f} "
					f"||grad_syn||={grad_syn_norm:.4e} "
					f"loss_real={loss_real.item():.4f} loss_syn={loss_syn.item():.4f}"
					)

		# Co-adaptation: a few optimizer steps of theta, trained on the *evolving* synthetic
		# buffer (detached: this branch must not reach back into the pixels), to keep theta
		# from staying frozen at its random initialization. This single-trajectory version differs
		# from the fully faithful DC algorithm (periodic random re-init of theta).
		if (outer_it + 1) % cfg.inner_steps_every_n == 0:
			for _ in range(cfg.inner_steps_count):
				B_syn_train = sample_synthetic_batch(cfg.matching_batch_size, detach=True)
				inner_optimizer.zero_grad(set_to_none=True)
				loss_inner = ctc_loss_for_batch(B_syn_train, inner_model, loss_function, device)
				loss_inner.backward()
				inner_optimizer.step()

	save_buffer(cfg.buffer_path)
	if skip_nonfinite_syn_grad:
		print(f"Skipped pixel updates (non-finite gradient): {n_skipped_syn}/{cfg.outer_iterations} "
				f"({100 * n_skipped_syn / max(cfg.outer_iterations, 1):.2f}%)")

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
