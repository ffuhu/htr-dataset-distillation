from torch import Tensor
import torch

def differentiable_ctc_loss(
		log_probs: Tensor,
		targets: Tensor,
		input_lengths: Tensor,
		target_lengths: Tensor,
		blank: int = 0,
		reduction: str = "mean",
		zero_infinity: bool = False,
		) -> Tensor:
	"""
	CTC loss (Graves et al., 2006) implemented as a composition of ordinary differentiable
	PyTorch ops (indexing, `logsumexp`) instead of the fused `torch.nn.CTCLoss` kernel.

	`torch.nn.CTCLoss` implements its own backward() in C++ and has no formula registered
	for *its own* derivative, so `torch.autograd.grad(loss, params, create_graph=True)`
	fails with "derivative for aten::_ctc_loss_backward is not implemented" -- this blocks
	any second-order use (e.g. DC-style gradient matching). This implementation is built
	entirely from ops that PyTorch already knows how to differentiate twice, so ordinary
	`create_graph=True` works with no special-casing.

	Only used where the second derivative is actually needed (see distill_dc_ctc.py): it is
	a Python loop over time steps and is substantially slower than the built-in CTCLoss, so
	it should not replace it where a single backward suffices.

	Args:
	- log_probs: Tensor(T, B, V), log-softmax'd, same convention `torch.nn.CTCLoss` expects.
	- targets: Tensor(B, L), padded target sequences (only the first target_lengths[b]
	  entries of row b are read).
	- input_lengths: Tensor(B,)
	- target_lengths: Tensor(B,), assumed >= 1 (HTR transcriptions are never empty).
	- reduction: "none" | "sum" | "mean" (mean divides each sample's loss by its target
	  length before averaging over the batch, matching `torch.nn.CTCLoss`'s convention).
	- zero_infinity: if True, samples with an impossible alignment (T < L + r, r = number of
	  pairs of equal consecutive target tokens, each needing a blank frame between them)
	  contribute 0 to the loss and no gradient, instead of +inf. Feasibility is decided
	  explicitly from the lengths, not from the alpha recursion, because the recursion uses
	  `finfo.min` instead of -inf and so never produces an infinite loss by itself.

	Note: experiments run before explicit feasibility was added here relied on `isinf(loss)` alone, so
	an infeasible sample got a loss of ~finfo.max and was NOT zeroed, even with zero_infinity=True. A
	scan of all saved buffers found infeasible samples only in IAM wr8 buffers that inherited pool line
	2844 (T=43 < 51) or 4458 (T=61 < 63), one sample each: DC iam_dc_{diag,diag_orig,noclass}_k616 and
	e12_iam_dc_randinit_k{308,616}_s0 (effect not quantified); MTT iam_mtt_ctc_k{308,616}_s0 and
	x3_iam_mtt_k308_lr{1,10}_s0, where it showed up as skipped non-finite iterations (829/2000 at
	K=308); and the E13b DC row main_dc_iam_k616.
	"""
	if reduction not in ("none", "sum", "mean"):
		raise ValueError(f"Unknown reduction: '{reduction}' (expected 'none', 'sum' or 'mean').")

	T, B, _V = log_probs.shape
	L = targets.shape[1]

	neg_inf = torch.finfo(log_probs.dtype).min

	# Extended label sequence z = [blank, y_1, blank, y_2, ..., y_L, blank], length 2L+1.
	extended = targets.new_full((B, 2*L + 1), blank)
	if L > 0:
		extended[:, 1::2] = targets

	# Position s (odd, a real label) may be reached directly from s-2 only if the label
	# there differs from the one at s -- this is what forbids collapsing repeated symbols
	# (e.g. "oo") into a single occurrence during decoding.
	can_skip_from_2_back = torch.zeros((B, 2*L + 1), dtype=torch.bool, device=targets.device)
	if L >= 2:
		can_skip_from_2_back[:, 3::2] = extended[:, 3::2] != extended[:, 1:-2:2]

	# log_probs_ext[t, b, s] = log_probs[t, b, extended[b, s]]
	log_probs_ext = log_probs.gather(2, extended.unsqueeze(0).expand(T, -1, -1))

	alpha = log_probs.new_full((T, B, 2*L + 1), neg_inf)
	alpha[0, :, 0] = log_probs_ext[0, :, 0]
	if L > 0:
		alpha[0, :, 1] = log_probs_ext[0, :, 1]

	neg_inf_col = log_probs.new_full((B, 1), neg_inf)
	neg_inf_pair = log_probs.new_full((B, 2*L + 1), neg_inf)

	for t in range(1, T):
		prev = alpha[t - 1]

		stay = prev
		from_1_back = torch.cat([neg_inf_col, prev[:, :-1]], dim=1)
		from_2_back = torch.where(
				can_skip_from_2_back,
				torch.cat([neg_inf_col, neg_inf_col, prev[:, :-2]], dim=1),
				neg_inf_pair,
				)

		alpha[t] = log_probs_ext[t] + torch.logsumexp(torch.stack([stay, from_1_back, from_2_back]), dim=0)

	batch_index = torch.arange(B, device=targets.device)
	last_valid_t = (input_lengths - 1).clamp(min=0)
	final_alpha = alpha[last_valid_t, batch_index] # (B, 2L+1)

	end_position_label = (2*target_lengths).clamp(min=0)
	end_position_blank = (2*target_lengths - 1).clamp(min=0)
	end_values = torch.stack([
			final_alpha.gather(1, end_position_label.unsqueeze(1)).squeeze(1),
			final_alpha.gather(1, end_position_blank.unsqueeze(1)).squeeze(1),
			], dim=1)

	loss = -torch.logsumexp(end_values, dim=1) # (B,), per-sample negative log-likelihood

	# Feasible iff T >= L + r (see zero_infinity above); repeats counted only within each target's length.
	if L >= 2:
		is_repeat = targets[:, 1:] == targets[:, :-1]
		within_target = torch.arange(L - 1, device=targets.device) < (target_lengths - 1).unsqueeze(1)
		n_repeats = (is_repeat & within_target).sum(dim=1)
	else:
		n_repeats = torch.zeros_like(target_lengths)
	feasible = input_lengths >= target_lengths + n_repeats

	loss = torch.where(feasible, loss, torch.full_like(loss, 0.0 if zero_infinity else float("inf")))

	if reduction == "none":
		return loss
	if reduction == "sum":
		return loss.sum()
	return (loss / target_lengths.clamp(min=1)).mean()
