import os
import platform
import re

if platform.system() == "Darwin":
	os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import random
from functools import partial

import torch
from torch import Tensor
from torch.func import functional_call

from omegaconf import DictConfig, OmegaConf
from hydra import main
from hydra.utils import instantiate

from lightning.pytorch import seed_everything

from data.TranscriptionCTC import to_tensordict
from code_version import stamp
from utils import differentiable_ctc_loss, get_device

def load_config(filename: str):
	return OmegaConf.load(filename)

EPOCH_RE = re.compile(r"epoch_(\d+)\.pt$")

class ExpertTrajectories:
	"""
	Reads the expert trajectory buffer that `callbacks.ExpertSnapshotter` saved during
	`train_expert_wr8.yaml` (one folder per seed, one `epoch_NNNN.pt` -- dict
	{param_name: Tensor} -- per epoch).

	Nothing is loaded into memory until a specific epoch is requested: each file is small
	(~0.75MB, 187K params), but a buffer of 15-20 experts x ~140 epochs is thousands of
	files and there is no need to hold them all at once.
	"""

	def __init__(self, expert_dir: str):
		self.expert_dir = expert_dir
		self.experts: dict[str, list[int]] = {}
		for name in sorted(os.listdir(expert_dir)):
			subdir = os.path.join(expert_dir, name)
			if not os.path.isdir(subdir):
				continue
			epochs = sorted(
					int(m.group(1))
					for f in os.listdir(subdir)
					if (m := EPOCH_RE.search(f)) is not None
					)
			if epochs:
				self.experts[name] = epochs

		assert self.experts, f"No expert trajectories found under '{expert_dir}' (expected <expert_dir>/<name>/epoch_NNNN.pt)."
		print(f"Expert buffer: {len(self.experts)} trajectories from '{expert_dir}', "
				f"epoch counts {[len(e) for e in self.experts.values()]}")

	def load(self, expert_name: str, epoch: int) -> dict[str, Tensor]:
		path = os.path.join(self.expert_dir, expert_name, f"epoch_{epoch:04d}.pt")
		return torch.load(path, map_location="cpu")

	def sample_start_target(self, rng: random.Random, max_start_epoch: int, epochs_ahead: int):
		"""
		Returns (expert_name, start_epoch, target_epoch), retrying with a fresh expert if
		the sampled one doesn't have an epoch >= epochs_ahead past the start (early-stopped
		short, e.g. a seed that converged/diverged sooner than the others).
		"""
		for _ in range(100):
			expert_name = rng.choice(list(self.experts.keys()))
			epochs = self.experts[expert_name]
			candidates = [e for e in epochs if e <= max_start_epoch]
			if not candidates:
				continue
			start_epoch = rng.choice(candidates)
			target_epoch = start_epoch + epochs_ahead
			if target_epoch in set(epochs):
				return expert_name, start_epoch, target_epoch
		raise RuntimeError(
				f"Could not find a valid (start, start+{epochs_ahead}) pair within {max_start_epoch} "
				f"epochs across {len(self.experts)} experts after 100 tries -- check expert_epochs_ahead "
				f"against the shortest trajectory ({min(len(e) for e in self.experts.values())} epochs).")

def ctc_loss_for_functional_model(model_shell, params, X, W, Y, L, loss_function, device) -> Tensor:
	logits, *_ = functional_call(model_shell, params, (X,))
	logits = logits.permute(1, 0, 2).log_softmax(dim=-1)
	input_lengths = W.div(model_shell.encoder.width_reduction, rounding_mode="trunc")
	return loss_function(logits, Y, input_lengths, L)

@main(version_base=None, config_path="../configs", config_name="distill_iam_mtt_ctc.yaml")
def _main(cfg: DictConfig):
	stamp()  # code commit; fails if it does not match CODE_VERSION
	if "seed" in cfg:
		seed_everything(cfg.seed)
	rng = random.Random(cfg.seed)

	# Same reasoning as distill_dc_ctc.py: the unrolled inner loop needs create_graph=True
	# all the way through, which MPS's max_pool2d backward doesn't support a second time.
	device = get_device(require_double_backward=True)
	print(f"Device: {device}")

	real_dataset = instantiate(cfg.real_dataset)
	print(f"Real (filtered) pool for buffer initialization: {len(real_dataset)} samples")

	cfg.max_h = real_dataset.stats["max_h"][0]
	cfg.max_w = real_dataset.stats["max_w"][0]
	cfg.max_l = real_dataset.stats["max_l"][0]
	cfg.t2i = real_dataset.class_to_int
	cfg.i2t = real_dataset.int_to_class
	cfg.vocabulary_size = len(cfg.i2t)

	loss_function = instantiate(cfg.loss_function)
	synthetic_loss_function = partial(
			differentiable_ctc_loss,
			blank=cfg.t2i["<CTC>"],
			reduction="mean",
			zero_infinity=True,
			)

	# Shape template only: its OWN parameters are never trained directly. Every forward
	# pass in this script substitutes a specific {name: Tensor} dict via functional_call
	# (either an expert snapshot, or the evolving student params in the inner loop) --
	# the model's own state_dict is irrelevant and never read after construction. Must
	# match the expert snapshots' architecture exactly (models@inner_model = the same
	# wr8 config used by configs/train_expert_wr8.yaml); parameter SHAPES depend only on
	# max_h (fixed_height=64 everywhere in this pipeline) and vocabulary_size, not on
	# max_w/max_l, so a mismatched real_dataset pool does not silently corrupt shapes --
	# but the architecture config group must still match, checked implicitly by
	# functional_call raising on any name/shape mismatch (fail loud).
	model_shell = instantiate(cfg.inner_model).to(device)
	model_shell.train() # BatchNorm uses batch stats during the inner loop, same as DC/DM's inner_model
	print(model_shell)
	print(f"Student/expert parameters: {sum(p.numel() for p in model_shell.parameters()):,}")

	experts = ExpertTrajectories(cfg.expert_dir)

	# --- Initialize the synthetic buffer: same convention as DC/DM (pixels-as-parameters,
	# fixed inherited labels, fixed per-sample width). cluster_file, if given, only controls
	# WHERE the buffer initializes from (medoids) -- MTT has no per-cluster matching loop
	# to condition (the "real" target is the expert trajectory, already dataset-wide), so
	# DC/DM's conditional_matching/cluster_members machinery has no MTT analogue and is
	# deliberately not ported here.
	if cfg.get("cluster_file"):
		cluster_cfg = load_config(cfg.cluster_file)
		assert int(cluster_cfg.k) == int(cfg.K), \
			f"cluster_file has k={cluster_cfg.k} but K={cfg.K} -- one synthetic sample per medoid."
		source_indices = [int(m) for m in cluster_cfg.medoids]
		print(f"Medoid-initialized buffer: {cfg.K} synthetics from '{cfg.cluster_file}'.")
	else:
		source_indices = rng.sample(range(len(real_dataset)), cfg.K)

	synthetic_images: list[torch.nn.Parameter] = []
	synthetic_labels: list[Tensor] = []
	for idx in source_indices:
		sample = real_dataset[idx]
		synthetic_images.append(torch.nn.Parameter(sample["img"].clone()))
		synthetic_labels.append(sample["label"].clone())

	def synthetic_batch(batch_size: int):
		# Mini-batch per inner step, NOT the whole K-buffer at once (unlike DC/DM's
		# batch_size=K convention). MTT keeps every inner step's activations alive
		# simultaneously (create_graph=True chains them into one graph for the final
		# trajectory-matching backward) -- a full K=100 batch x inner_steps=20 unroll
		# was measured to OOM a 20GB GPU on the very first step (empirically: a single
		# K=100 forward/backward is ~5.3GB; 20 of those retained at once vastly exceeds
		# available memory). cfg.inner_batch_size defaults to the SAME batch size the
		# expert trajectories themselves were trained with (8, see train_expert_wr8.yaml)
		# -- not just a memory fix: it makes one synthetic step comparable in scale to
		# one real step, which is what inner_steps replicating expert_epochs_ahead real
		# epochs is actually supposed to mean.
		idxs = rng.sample(range(len(synthetic_images)), min(batch_size, len(synthetic_images)))
		items = [to_tensordict(synthetic_images[i], synthetic_labels[i], "synthetic") for i in idxs]
		return real_dataset.collate_fn(items)

	# MTT's learnable per-step synthetic learning rate (Cazenavette et al. 2022): a SCALAR
	# meta-parameter optimized jointly with the pixels via the same trajectory-matching
	# gradient, not the pixel optimizer's own lr. Named distinctly from DC/DM's `syn_lr`
	# (which is cfg.pixel_lr here) to avoid conflating two different learning rates that
	# happen to share a name across papers.
	inner_lr = torch.nn.Parameter(torch.tensor(float(cfg.mtt_inner_lr_init), device=device))

	syn_optimizer = torch.optim.Adam([
			{"params": synthetic_images, "lr": cfg.pixel_lr},
			{"params": [inner_lr], "lr": cfg.mtt_inner_lr_lr},
			])

	for outer_it in range(cfg.outer_iterations):
		expert_name, start_epoch, target_epoch = experts.sample_start_target(
				rng, cfg.max_start_epoch, cfg.expert_epochs_ahead)
		expert_start = experts.load(expert_name, start_epoch)
		expert_target = experts.load(expert_name, target_epoch)

		assert expert_start.keys() == expert_target.keys(), \
			"Expert start/target snapshots have different parameter names -- architecture mismatch?"

		# requires_grad_ only on this initial leaf: every subsequent student_params[name] is
		# produced by arithmetic on this graph and inherits requires_grad automatically.
		student_params = {name: t.to(device).clone().requires_grad_(True) for name, t in expert_start.items()}

		with torch.backends.cudnn.flags(enabled=False): # LSTM has no cuDNN double-backward (see distill_dc_ctc.py)
			for _ in range(cfg.inner_steps):
				batch = synthetic_batch(cfg.inner_batch_size).to(device)
				loss_inner = ctc_loss_for_functional_model(
						model_shell, student_params, batch["img"], batch["shape"][:, 1], batch["label"], batch["length"],
						synthetic_loss_function, device)
				grads = torch.autograd.grad(loss_inner, list(student_params.values()), create_graph=True)
				student_params = {
						name: p - inner_lr * g
						for (name, p), g in zip(student_params.items(), grads)
						}

		student_flat = torch.cat([student_params[name].reshape(-1) for name in sorted(student_params)])
		target_flat = torch.cat([expert_target[name].to(device).reshape(-1) for name in sorted(expert_target)])
		start_flat = torch.cat([expert_start[name].to(device).reshape(-1) for name in sorted(expert_start)])

		numerator = (student_flat - target_flat).pow(2).sum()
		denominator = (start_flat - target_flat).pow(2).sum().detach().clamp(min=1e-12)
		matching_loss = numerator / denominator

		syn_optimizer.zero_grad(set_to_none=True)
		matching_loss.backward()
		# Guard against the failure mode found on GW and later confirmed on LAM (buffers
		# with up to ~15% NaN pixels at K=991):
		# inner_lr is an unconstrained learnable scalar, and an unlucky (expert, start,
		# target) sample combined with an already-large inner_lr can make the inner-loop
		# SGD step `p - inner_lr * g` overshoot far enough that the CTC loss goes
		# non-finite -- once that happens, backward() produces a non-finite gradient, and
		# stepping the Adam optimizer on it permanently corrupts its moving averages
		# (every later iteration logs nan forever after, silently wasting the rest of the
		# run). The original guard only checked `matching_loss` itself, but a finite loss
		# does not guarantee finite gradients (CTC backward can diverge asymmetrically
		# from its forward value) -- when that gap was hit, clip_grad_norm_ computed a NaN
		# total norm from the NaN gradient, which scales EVERY parameter's gradient (not
		# just the offending one) to NaN, and the following optimizer.step() permanently
		# wrote NaN into whichever synthetic pixels happened to be in that iteration's
		# sampled mini-batches (LAM's investigation confirmed only the images touched that
		# iteration end up corrupted, not the whole buffer -- consistent with this path).
		# Checking gradient finiteness explicitly closes that gap.
		grads_finite = all(
				torch.isfinite(p.grad).all()
				for p in synthetic_images + [inner_lr]
				if p.grad is not None
				)
		if not torch.isfinite(matching_loss) or not grads_finite:
			print(f"[WARN] [{outer_it:4d}/{cfg.outer_iterations}] non-finite matching_loss or "
					f"gradient (expert={expert_name} start={start_epoch} target={target_epoch}) "
					"-- skipping update.")
			continue
		torch.nn.utils.clip_grad_norm_(synthetic_images + [inner_lr], max_norm=10.0)
		syn_optimizer.step()
		inner_lr.data.clamp_(min=1e-4, max=cfg.mtt_inner_lr_max)  # E13: null = no upper bound (original MTT)

		if outer_it % cfg.log_every == 0:
			print(
					f"[{outer_it:4d}/{cfg.outer_iterations}] expert={expert_name} "
					f"start={start_epoch} target={target_epoch} "
					f"matching_loss={matching_loss.item():.4f} "
					f"student-target={numerator.sqrt().item():.4f} "
					f"expert-displacement={denominator.sqrt().item():.4f} "
					f"inner_lr={inner_lr.item():.4e} "
					f"loss_inner(last step)={loss_inner.item():.4f}"
					)

	buffer = {
			"images": [p.detach().clone() for p in synthetic_images],
			"labels": [label.clone() for label in synthetic_labels],
			"source_indices": source_indices,
			}

	buffer_dir = os.path.dirname(cfg.buffer_path)
	if buffer_dir:
		os.makedirs(buffer_dir, exist_ok=True)
	torch.save(buffer, cfg.buffer_path)
	print(f"Saved MTT-distilled buffer ({cfg.K} samples) to '{cfg.buffer_path}'. Final inner_lr={inner_lr.item():.4e}")

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
