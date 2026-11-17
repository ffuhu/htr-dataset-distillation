"""
E13 (pre-registered): is the MTT objective reachable?

From snapshot k of an expert, a student trained on REAL LINES (no augmentation) tries to reach snapshot
k+ahead. If not even real data can do it, the MTT loss cannot decrease with synthetic data and the failure lies
in the objective, not in the pixel optimisation. Measures, with L = ||theta - theta_tgt||^2 / ||theta_k - theta_tgt||^2
(the same normalisation as distill_mtt_ctc.py):
  A) MTT inner loop with real lines: `inner_steps` SGD steps, batch `inner_batch`, alpha on a grid
     -> G0 gate statistic: best L over the grid, median over (expert, start)
  C) the expert's own recipe (optimizer, lr and batch), fresh state, as many steps as separate the two snapshots
  D) cosine between -mean gradient (`grad_batches` batches) and the expert's displacement
Writes one JSON line per (expert, start) to --out.

Starts in EPOCHS (--start-epochs); with snapshots every M steps (--steps-per-snapshot M) they are converted to the
snapshot index round(epoch * steps_per_epoch / M), with steps_per_epoch = ceil(N / expert_batch).
"""
import argparse
import json
import math
import os
import random

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from code_version import stamp
from distill_mtt_ctc import ExpertTrajectories, ctc_loss_for_functional_model, load_config

p = argparse.ArgumentParser()
p.add_argument("--distill-config", required=True)  # e.g. distill_iam_mtt_ctc_wr8: gives the real pool and the architecture
p.add_argument("--expert-dir", required=True)
p.add_argument("--experts", nargs="+", required=True)
p.add_argument("--start-epochs", nargs="+", type=int, required=True)
p.add_argument("--steps-per-snapshot", type=int, default=None)  # None = one snapshot per epoch (Adam experts)
p.add_argument("--snap-ahead", type=int, required=True)
p.add_argument("--expert-opt", choices=["adam", "sgd"], required=True)
p.add_argument("--expert-lr", type=float, required=True)
p.add_argument("--expert-batch", type=int, required=True)
p.add_argument("--inner-steps", type=int, required=True)
p.add_argument("--inner-batch", type=int, required=True)
p.add_argument("--alphas", nargs="+", type=float, required=True)
p.add_argument("--grad-batches", type=int, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--out", required=True)
args = p.parse_args()
print(f"args: {vars(args)}", flush=True)
code_commit = stamp()

for name, fn in [("int_div", lambda x, y: x // y), ("mul", lambda x, y: x * y),
				("eval", lambda x: eval(x)), ("load", load_config)]:
	OmegaConf.register_new_resolver(name, fn)
with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
	cfg = compose(config_name=args.distill_config, overrides=[f"expert_dir={args.expert_dir}"])
real = instantiate(cfg.real_dataset)
with open_dict(cfg):
	cfg.max_h = real.stats["max_h"][0]
	cfg.max_w = real.stats["max_w"][0]
	cfg.max_l = real.stats["max_l"][0]
	cfg.t2i = real.class_to_int
	cfg.i2t = real.int_to_class
	cfg.vocabulary_size = len(cfg.i2t)
device = torch.device("cuda")
shell = instantiate(cfg.inner_model).to(device).train()  # same mode as the MTT inner loop
loss_function = instantiate(cfg.loss_function)  # the CTC the experts are trained with
N = len(real)
steps_per_epoch = math.ceil(N / args.expert_batch)
steps_between = args.snap_ahead * (args.steps_per_snapshot or steps_per_epoch)
print(f"pool N={N}, expert steps/epoch={steps_per_epoch}, expert steps between start and target={steps_between}", flush=True)
experts = ExpertTrajectories(args.expert_dir)

def batch(rng: random.Random, size: int):
	return real.collate_fn([real[i] for i in rng.sample(range(N), size)]).to(device)

def loss_on(params, b):
	return ctc_loss_for_functional_model(shell, params, b["img"], b["shape"][:, 1], b["label"], b["length"], loss_function, device)

def flat(d):
	return torch.cat([d[k].reshape(-1) for k in sorted(d)])

def train_from(start, target, opt_name, lr, steps, batch_size, seed):
	"""Returns (L, |ds|/|de|, cos(ds, de)); None if the loss or the parameters become non-finite."""
	rng = random.Random(seed)
	params = {k: v.clone().requires_grad_(True) for k, v in start.items()}
	opt = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[opt_name](list(params.values()), lr=lr)
	for _ in range(steps):
		opt.zero_grad(set_to_none=True)
		loss = loss_on(params, batch(rng, batch_size))
		if not torch.isfinite(loss):
			return None
		loss.backward()
		opt.step()
	with torch.no_grad():
		s, t, s0 = flat(params), flat(target), flat(start)
		if not torch.isfinite(s).all():
			return None
		de, ds = t - s0, s - s0
		return ((s - t).pow(2).sum() / de.pow(2).sum()).item(), (ds.norm() / de.norm()).item(), \
			torch.nn.functional.cosine_similarity(ds, de, dim=0).item()

with open(args.out, "a") as out:
	for name in args.experts:
		available = experts.experts[name]
		for epoch in args.start_epochs:
			k = epoch if args.steps_per_snapshot is None else round(epoch * steps_per_epoch / args.steps_per_snapshot)
			if k + args.snap_ahead not in available:
				print(f"{name} epoch={epoch}: snapshot {k + args.snap_ahead} missing (have {available[-1]}), skipped")
				continue
			start = {n: v.to(device) for n, v in experts.load(name, k).items()}
			target = {n: v.to(device) for n, v in experts.load(name, k + args.snap_ahead).items()}
			de = flat(target) - flat(start)

			rng = random.Random(args.seed)
			params = {n: v.clone().requires_grad_(True) for n, v in start.items()}
			g_mean = {n: torch.zeros_like(v) for n, v in start.items()}
			for _ in range(args.grad_batches):
				grads = torch.autograd.grad(loss_on(params, batch(rng, args.inner_batch)), list(params.values()))
				for n, g in zip(params, grads):
					g_mean[n] += g / args.grad_batches
			cos_g = torch.nn.functional.cosine_similarity(-flat(g_mean), de, dim=0).item()

			inner = {a: train_from(start, target, "sgd", a, args.inner_steps, args.inner_batch, args.seed) for a in args.alphas}
			finite = [r[0] for r in inner.values() if r is not None]
			best = min(finite) if finite else None
			replay = train_from(start, target, args.expert_opt, args.expert_lr, steps_between, args.expert_batch, args.seed)

			rec = {"code_commit": code_commit, "expert_dir": args.expert_dir, "expert": name, "start_epoch": epoch,
				"snapshot": k, "snap_ahead": args.snap_ahead, "steps_between": steps_between,
				"expert_displacement": de.norm().item(), "cos_neg_grad": cos_g,
				"inner": {str(a): r for a, r in inner.items()}, "best_inner_L": best, "replay": replay}
			out.write(json.dumps(rec) + "\n")
			out.flush()
			print(f"{name} epoch={epoch} snap={k}->{k + args.snap_ahead} |de|={de.norm():.3f} cos(-g,de)={cos_g:.4f} "
				f"best inner L={best} replay(L,|ds|/|de|,cos)={replay}", flush=True)
			for a, r in inner.items():
				print(f"    alpha={a:<6} (L,|ds|/|de|,cos)={r}", flush=True)
