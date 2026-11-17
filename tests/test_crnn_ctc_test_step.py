"""
Regression test for the test-CER bug: `CRNN_CTC_Wrapper.test_step` used to compare the prediction
with `Y[:L-1]` (the reference without its last character) instead of `Y[:L]` as in validation.

A fake model whose outputs decode EXACTLY to the reference must give distance 0 and length = sum of L.
Usage (from the repository root, no GPU needed):  python tests/test_crnn_ctc_test_step.py
"""
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch
from tensordict import TensorDict

sys.path.insert(0, "src")
from models.CRNN_CTC_Wrapper import CRNN_CTC_Wrapper  # noqa: E402

BLANK, WR, V = 1, 8, 6  # <CTC> = 1 as in the project vocabularies; width_reduction 8


def perfect_logits(labels: list[list[int]], T: int) -> torch.Tensor:
	"""One-hot logits whose greedy CTC decoding is exactly each label (symbol, blank, symbol, ...)."""
	out = torch.full((len(labels), T, V), -10.0)
	out[:, :, BLANK] = 10.0
	for b, lab in enumerate(labels):
		for i, tok in enumerate(lab):
			out[b, 2 * i, BLANK] = -10.0
			out[b, 2 * i, tok] = 10.0
	return out


def main():
	labels = [[2, 3, 4], [5, 2, 2, 3]]  # includes a repeated character (separated by a blank in the logits)
	T = 12
	L = torch.tensor([len(x) for x in labels])
	Y = torch.zeros(len(labels), int(L.max()), dtype=torch.long)
	for b, lab in enumerate(labels):
		Y[b, :len(lab)] = torch.tensor(lab)
	batch = TensorDict({"img": torch.zeros(len(labels), 1, 64, T * WR), "label": Y, "length": L,
						"shape": torch.tensor([[64, T * WR]] * len(labels))}, batch_size=[len(labels)])
	batch["split"] = ["test"] * len(labels)

	logits = perfect_logits(labels, T)
	w = CRNN_CTC_Wrapper.__new__(CRNN_CTC_Wrapper)
	torch.nn.Module.__init__(w)
	w.__dict__["model"] = SimpleNamespace(predict=lambda X: (logits,), encoder=SimpleNamespace(width_reduction=WR))
	w.t2i = {"<PAD>": 0, "<CTC>": BLANK}
	w.i2t = {i: chr(ord("a") + i) for i in range(V)}
	w.test_sample_id, w.total_symbol_distance, w.total_symbol_length = defaultdict(int), defaultdict(int), defaultdict(int)
	w._fabric = None  # LightningModule.logger reads _fabric and then _trainer.logger
	w._trainer = SimpleNamespace(logger=SimpleNamespace(log_metrics=lambda m: None), loggers=[])
	CRNN_CTC_Wrapper.test_step(w, batch, 0)

	dist, length = w.total_symbol_distance["test"], w.total_symbol_length["test"]
	print(f"distancia={dist} longitud={length} (esperado 0 y {int(L.sum())})")
	assert length == int(L.sum()) and dist == 0, "test_step no compara contra la referencia completa"
	print("OK")


if __name__ == "__main__":
	main()
