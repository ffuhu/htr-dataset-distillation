"""
Regression test for `differentiable_ctc_loss` with impossible alignments (T < L + r, r = pairs of
equal consecutive tokens): the recursion uses `finfo.min` instead of -inf, so without explicit feasibility the
loss comes out ~3.4e38 (finite), `zero_infinity=True` does not zero it, and `zero_infinity=False` does not
return +inf as `torch.nn.functional.ctc_loss` does.

Cases at the exact boundary T = L + r (feasible) and T = L + r - 1 (impossible), with and without repeats.
Usage (from the repository root, no GPU needed):  python tests/test_differentiable_ctc_infeasible.py
"""
import sys

import torch
from torch.nn.functional import ctc_loss as torch_ctc_loss

sys.path.insert(0, "src")
from utils import differentiable_ctc_loss  # noqa: E402

BLANK, V = 0, 6

# (target, T): viable sii T >= len(target) + repeticiones consecutivas
CASES = [
		([1, 2, 3], 3),         # L+r = 3: viable
		([1, 2, 3], 2),         # imposible
		([1, 2, 2, 3], 5),      # L+r = 5: viable
		([1, 2, 2, 3], 4),  # impossible: T >= L but not T >= L + r
		([4, 4, 4], 5),         # L+r = 5: viable
		([4, 4, 4], 4),         # imposible
		([5], 1),               # L+r = 1: viable
		]
FEASIBLE = torch.tensor([T >= len(y) + sum(a == b for a, b in zip(y, y[1:])) for y, T in CASES])


def make_batch():
	generator = torch.Generator().manual_seed(0)
	B, Tmax, Lmax = len(CASES), max(T for _, T in CASES), max(len(y) for y, _ in CASES)
	log_probs = torch.randn(Tmax, B, V, dtype=torch.float64, generator=generator).log_softmax(-1)
	targets = torch.zeros(B, Lmax, dtype=torch.long)
	for b, (y, _) in enumerate(CASES):
		targets[b, :len(y)] = torch.tensor(y)
	input_lengths = torch.tensor([T for _, T in CASES])
	target_lengths = torch.tensor([len(y) for y, _ in CASES])
	return log_probs, targets, input_lengths, target_lengths


def main() -> bool:
	all_ok = True
	log_probs, targets, input_lengths, target_lengths = make_batch()

	for zero_infinity in (False, True):
		ref = torch_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=BLANK, reduction="none", zero_infinity=zero_infinity)
		mine = differentiable_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=BLANK, reduction="none", zero_infinity=zero_infinity)
		assert torch.isfinite(ref).equal(FEASIBLE) or zero_infinity, "el caso de prueba no es lo que dice ser"
		ok = torch.allclose(ref, mine, atol=1e-6, rtol=1e-4)  # allclose trata inf == inf
		all_ok &= ok
		print(f"zero_infinity={zero_infinity}: torch={ref.tolist()}\n{'':20s}mine ={mine.tolist()}  {'OK' if ok else 'FAIL'}")

	# zero_infinity=True: impossible samples must contribute no gradient; feasible ones, a finite gradient.
	lp = log_probs.clone().requires_grad_(True)
	differentiable_ctc_loss(lp, targets, input_lengths, target_lengths, blank=BLANK, reduction="sum", zero_infinity=True).backward()
	ok_finite = bool(torch.isfinite(lp.grad).all())
	ok_zero = bool((lp.grad[:, ~FEASIBLE] == 0).all())
	all_ok &= ok_finite and ok_zero
	print(f"grad finito: {'OK' if ok_finite else 'FAIL'}; grad nulo en imposibles: {'OK' if ok_zero else 'FAIL'}")
	return all_ok


if __name__ == "__main__":
	if not main():
		print("FAIL")
		raise SystemExit(1)
	print("OK")
