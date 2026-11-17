import torch

def get_device(require_double_backward: bool = False) -> torch.device:
	"""
	CUDA > MPS (Apple GPU) > CPU. Two known MPS gaps this deliberately works around:

	- torch.nn.CTCLoss has no MPS kernel at all; with PYTORCH_ENABLE_MPS_FALLBACK=1 (set for
	  Darwin in these scripts) that op silently falls back to CPU per-call instead of erroring,
	  while the rest of the model (conv/BLSTM/etc.) still runs on MPS. Fine for ordinary
	  (first-order) training -- e.g. distill_dm_ctc.py.
	- MPS's max_pool2d backward is not itself differentiable a second time ("not infinitely
	  differentiable" unless return_indices=True), which breaks the create_graph=True path
	  DC-CTC needs to backprop the matching loss through grad_syn into the synthetic pixels.
	  Pass require_double_backward=True (distill_dc_ctc.py) to skip MPS for that reason and
	  fall back to CPU instead.
	"""
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available() and not require_double_backward:
		return torch.device("mps")
	return torch.device("cpu")
