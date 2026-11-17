"""
E13b: do DM and DC reduce their own objective? The DM/DC analogue of the MTT check in E13.

The distillation logs do not tell: each iteration logs the loss with a new random network and a new real batch
(and DC re-initialises the network every `reinit_every_n`), so 40 noisy points cannot separate trend from noise.
Here the objective of a run is evaluated with its INITIAL and its FINAL buffer under the SAME networks and real
batches (paired comparison), with the Hydra config saved by that run and in the same mode it was optimised in:
  - conditional: one synthetic line against a batch of its pseudo-class
  - round_robin_syn: one synthetic line against a global real batch
  - batch mode: `matching_batch_size` random synthetic lines against a global real batch
DM: distance between mean embeddings with a frozen random encoder (as in `distill_dm_ctc.py`).
DC: distance between CTC gradients (real: CTCLoss; synthetic: differentiable CTC, as in `distill_dc_ctc.py`)
on a freshly initialised network (the point of each DC re-initialisation) or, with --coadapt-steps S > 0, on that
network after S Adam steps (lr = the run's inner_lr, batches of matching_batch_size lines) on the FINAL buffer: the
states DC visits within each re-initialisation cycle (5 steps every 10 iterations, up to ~100 per cycle of 200). The
same network is used for the initial and the final buffer.

Output: one JSON line with means, final/initial ratio and paired mean of log(final/initial) with 95% CI.
"""
import argparse
import json
import math
import os
import random
import statistics
from functools import partial

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from code_version import stamp
from data.TranscriptionCTC import to_tensordict
from distill_dc_ctc import ctc_loss_for_batch, dc_original_matching_loss, gradient_matching_loss
from distill_dm_ctc import freeze, load_config, pool_features
from utils import differentiable_ctc_loss

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["dm", "dc"], required=True)
p.add_argument("--run-config", required=True)  # <run>/.hydra/config.yaml of the distillation run
p.add_argument("--init", required=True)
p.add_argument("--final", required=True)
p.add_argument("--units", type=int, required=True)  # lines (per-line modes) or synthetic batches (batch mode)
p.add_argument("--draws", type=int, required=True)  # networks per unit
p.add_argument("--seed", type=int, required=True)
p.add_argument("--coadapt-steps", type=int, required=True)  # DC only; 0 = freshly initialised network
p.add_argument("--out", required=True)
args = p.parse_args()
print(f"args: {vars(args)}", flush=True)
code_commit = stamp()

for name, fn in [("int_div", lambda x, y: x // y), ("mul", lambda x, y: x * y),
				("eval", lambda x: eval(x)), ("load", load_config)]:
	OmegaConf.register_new_resolver(name, fn)
cfg = OmegaConf.load(args.run_config)
real = instantiate(cfg.real_dataset)
cfg.max_h = real.stats["max_h"][0]
cfg.max_w = real.stats["max_w"][0]
cfg.max_l = real.stats["max_l"][0]
cfg.t2i = real.class_to_int
cfg.i2t = real.int_to_class
cfg.vocabulary_size = len(cfg.i2t)
device = torch.device("cuda")
N = len(real)

assert not cfg.get("dsa", False), "DSA runs not supported by this evaluator."
if args.method == "dm":
	assert cfg.get("encoder_source", "random") == "random", "Only random-encoder DM is evaluated."
	assert args.coadapt_steps == 0, "DM uses frozen random encoders: no co-adaptation."

conditional = bool(cfg.get("conditional_matching", cfg.get("cluster_file") is not None))
round_robin = bool(cfg.get("round_robin_syn", False))
mode = "conditional" if conditional else ("round_robin" if round_robin else "batch")
cluster_members = None
if conditional:
	assignments = [int(a) for a in load_config(cfg.cluster_file).clusters]
	assert len(assignments) == N, (len(assignments), N)
	cluster_members = [[i for i, a in enumerate(assignments) if a == c] for c in range(cfg.K)]

init, final = torch.load(args.init, map_location="cpu"), torch.load(args.final, map_location="cpu")
K = len(final["images"])
assert K == len(init["images"]) == cfg.K, (K, len(init["images"]), cfg.K)
assert list(init["source_indices"]) == list(final["source_indices"]), "init/final buffers start from different lines"
assert all(torch.equal(a, b) for a, b in zip(init["labels"], final["labels"])), "init/final labels differ"
for i in random.Random(0).sample(range(K), min(K, 20)):
	assert torch.equal(init["images"][i], real[int(init["source_indices"][i])]["img"]), \
		f"init buffer line {i} is not the real line it claims to start from"
drift = math.sqrt(sum(float((f - s).pow(2).sum()) for f, s in zip(final["images"], init["images"]))
				/ sum(float(s.pow(2).sum()) for s in init["images"]))
assert math.isfinite(drift), "non-finite pixels in a buffer"

def syn_batch(buffer, idxs):
	return real.collate_fn([to_tensordict(buffer["images"][i], buffer["labels"][i], "synthetic") for i in idxs]).to(device)

if args.method == "dc":
	loss_function = instantiate(cfg.loss_function)
	syn_loss_function = partial(differentiable_ctc_loss, blank=cfg.t2i["<CTC>"], reduction="mean", zero_infinity=True)
	distance = {"cosine": gradient_matching_loss, "dc_original": dc_original_matching_loss}[cfg.get("match_distance", "cosine")]

def objective(net, b_real, b_syn):
	if args.method == "dm":
		with torch.no_grad():
			wr = int(net.width_reduction)
			fr = pool_features(net(b_real["img"]), cfg.pooling, b_real["shape"][:, 1] // wr)
			fs = pool_features(net(b_syn["img"]), cfg.pooling, b_syn["shape"][:, 1] // wr)
			return float((fr.mean(dim=0) - fs.mean(dim=0)).pow(2).sum())
	params = list(net.parameters())
	g_real = torch.autograd.grad(ctc_loss_for_batch(b_real, net, loss_function, device), params)
	with torch.backends.cudnn.flags(enabled=False):
		g_syn = torch.autograd.grad(ctc_loss_for_batch(b_syn, net, syn_loss_function, device), params)
	return float(distance(g_real, g_syn))

rng = random.Random(args.seed)
units = []
for _ in range(args.units):
	if mode == "batch":
		units.append(rng.sample(range(K), min(cfg.matching_batch_size, K)))
	else:
		units.append([rng.randrange(K)])

def build_net(d):
	torch.manual_seed(args.seed * 100003 + d)
	if args.method == "dm":
		return freeze(instantiate(cfg.embedding_encoder)).to(device)
	net = instantiate(cfg.inner_model).to(device)
	if args.coadapt_steps:
		opt = torch.optim.Adam(net.parameters(), lr=cfg.inner_lr)
		r = random.Random(f"coadapt-{args.seed}-{d}")
		for _ in range(args.coadapt_steps):
			opt.zero_grad(set_to_none=True)
			ctc_loss_for_batch(syn_batch(final, r.sample(range(K), min(cfg.matching_batch_size, K))),
				net, loss_function, device).backward()
			opt.step()
	return net

nets = [build_net(d) for d in range(args.draws)]
per_unit = []  # (mean init, mean final) over draws
for u, syn_idxs in enumerate(units):
	li, lf = [], []
	for d in range(args.draws):
		net = nets[d]
		r = random.Random(f"{args.seed}-{u}-{d}")
		pool = cluster_members[syn_idxs[0]] if conditional else range(N)
		b_real = real.collate_fn([real[i] for i in r.sample(pool, min(cfg.matching_batch_size, len(pool)))]).to(device)
		# The DC network is in train mode (as during distillation) and has dropout: the same seed before each
		# evaluation gives the same mask to the initial and the final buffer (a truly paired comparison).
		torch.manual_seed(args.seed * 1000003 + u * 1009 + d)
		li.append(objective(net, b_real, syn_batch(init, syn_idxs)))
		torch.manual_seed(args.seed * 1000003 + u * 1009 + d)
		lf.append(objective(net, b_real, syn_batch(final, syn_idxs)))
	per_unit.append((statistics.mean(li), statistics.mean(lf)))

# A single-member pseudo-class (conditional) has the initial line itself as real batch: initial objective 0.
# Those units have no ratio; they are excluded from the paired ratio and counted.
degenerate = sum(i == 0.0 for i, _ in per_unit)
logr = [math.log(f / i) for i, f in per_unit if i > 0.0]
m, sd = statistics.mean(logr), statistics.stdev(logr)
half = 1.96 * sd / math.sqrt(len(logr))  # n units >= 50: normal instead of t
mean_i = statistics.mean(i for i, _ in per_unit)
mean_f = statistics.mean(f for _, f in per_unit)
rec = {"code_commit": code_commit, "method": args.method, "mode": mode, "run_config": args.run_config,
	"init": args.init, "final": args.final, "coadapt_steps": args.coadapt_steps, "K": K, "units": len(units), "draws": args.draws, "drift": drift,
	"mean_init": mean_i, "mean_final": mean_f, "ratio_of_means": mean_f / mean_i,
	"geo_ratio": math.exp(m), "geo_ratio_ci95": [math.exp(m - half), math.exp(m + half)],
	"frac_units_lower": sum(f < i for i, f in per_unit) / len(per_unit), "degenerate_units": degenerate}
with open(args.out, "a") as out:
	out.write(json.dumps(rec) + "\n")
print(json.dumps(rec), flush=True)
