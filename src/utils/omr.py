from typing import Sequence
from torch import Tensor

def greedy_decoding(logits: Tensor, ignore_index: int) -> Tensor:
	out_best = logits.argmax(dim=-1)
	if len(out_best.shape) > 1:
		out_best = out_best.unique_consecutive(dim=-1)
	else:
		out_best = out_best.unique_consecutive()

	return out_best[out_best != ignore_index]

def levenshtein(a: Sequence, b: Sequence) -> int:
	n, m = len(a), len(b)

	if n > m:
		a, b = b, a
		n, m = m, n
	current = list(range(n+1))
	previous = [0]*n
	for i in range(1, m+1):
		previous, current = current, [i] + [0]*n
		# previous, current = current, previous
		current[0] = i
		for j in range(1, n+1):
			add, delete = previous[j]+1, current[j-1]+1
			change = previous[j-1]

			if a[j-1] != b[i-1]:
				change += 1

			current[j] = min(add, delete, change)

	return current[n]

def SER(pred: Sequence, target: Sequence) -> tuple[int, int, float]:
	sed = levenshtein(pred, target)

	return sed, len(target), sed/len(target)
