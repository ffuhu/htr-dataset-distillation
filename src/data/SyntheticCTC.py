from typing import Callable
from torch import Tensor, long as torch_long, load as torch_load
from tensordict import TensorDict

from .base import MyDataset
from .TranscriptionCTC import to_tensordict

import yaml

class SyntheticCTCDataset(MyDataset):
	"""
	Wraps a distilled buffer (K synthetic images + fixed labels inherited from the real
	samples they were initialized from) so it can be plugged into `DataModule` exactly like
	`data.TranscriptionCTC.HFCTC` (same `TensorDict` layout, same collate_fn).

	The buffer file is expected to be a dict saved with `torch.save`:
		- "images": list[Tensor], each (channels, height, width_k), float, already in the
		  same preprocessed pixel space `HFCTC.__getitem__` produces (i.e. post `img_transform`).
		- "labels": list[Tensor], each (length_k,), long, token ids in the shared vocabulary.
	Width and target length are always derived from the tensors' own shapes, never stored as
	separate metadata, to avoid them drifting out of sync with the actual data.
	"""

	def __init__(
			self,
			collate_fn: Callable[[TensorDict], TensorDict],
			buffer_path: str,
			vocabulary_file: str,
			splits: list[str] = ["synthetic"],
			batch_size: int = 1,
			shuffle: bool = False,
			num_workers: int = 0,
			persistent_workers: bool = False,
			img_transform=None,
			*args,
			**kwargs
			) -> None:
		super().__init__(
				collate_fn,
				batch_size=batch_size,
				shuffle=shuffle,
				num_workers=num_workers,
				persistent_workers=persistent_workers,
				*args,
				**kwargs
				)

		self.buffer_path: str = buffer_path
		self.splits: list[str] = splits
		self.splits_str: str = "_".join(self.splits)

		buffer = torch_load(buffer_path, map_location="cpu")

		self.images: list[Tensor] = list(buffer["images"])
		self.labels: list[Tensor] = [l.to(torch_long) for l in buffer["labels"]]
		self.source_indices: list[int] = list(buffer.get("source_indices", range(len(self.images))))

		# Optional per-frame teacher logits (KD): list[Tensor (T_max, V)], pre-padded
		# to a common T_max by the relabelling script so the generic collate_fn can stack
		# them. Valid frames are always derived downstream from the image width (see
		# CRNN_CTC_KD_Wrapper), so the padding never leaks into the loss.
		self.soft_labels: list[Tensor] | None = list(buffer["soft_labels"]) if "soft_labels" in buffer else None

		assert len(self.images) == len(self.labels), \
			f"Buffer at '{buffer_path}' has {len(self.images)} images but {len(self.labels)} labels."
		if self.soft_labels is not None:
			assert len(self.soft_labels) == len(self.images), \
				f"Buffer at '{buffer_path}' has {len(self.images)} images but {len(self.soft_labels)} soft_labels."

		# Tensor-native augmentation (data.augmentation.full_pipeline_tensor), to restore the
		# augmentation symmetry with the real-line baselines (RandomAffine is worth
		# ~15-16 CER points at wr8/K=100). Hard-label buffers only: augmenting an image whose
		# per-frame soft labels were computed on the un-augmented pixels would silently
		# desynchronize image and supervision.
		self.img_transform = img_transform if (img_transform is None or len(img_transform) > 0) else None
		assert not (self.img_transform is not None and self.soft_labels is not None), \
			f"Buffer at '{buffer_path}' carries soft_labels: augmentation would desynchronize them."

		self.load_vocabulary(vocabulary_file)
		self.stats = self.compute_stats()

	def load_vocabulary(self, vocabulary_file: str):
		vocabulary_file = vocabulary_file.replace(".yaml", "").replace("-c2i", "").replace("-i2c", "")

		with open(f"data/{vocabulary_file}-c2i.yaml", "r") as file:
			self.class_to_int = yaml.safe_load(file)
		with open(f"data/{vocabulary_file}-i2c.yaml", "r") as file:
			self.int_to_class = yaml.safe_load(file)

	def compute_stats(self) -> dict[str, tuple[int | float, list[str]]]:
		heights = [int(img.shape[1]) for img in self.images]
		widths = [int(img.shape[2]) for img in self.images]
		lengths = [int(label.shape[0]) for label in self.labels]
		n = len(self.images)

		return {
				"max_h": (max(heights), []),
				"max_w": (max(widths), []),
				"max_l": (max(lengths), []),
				"min_h": (min(heights), []),
				"min_w": (min(widths), []),
				"min_l": (min(lengths), []),
				"mean_h": (sum(heights) / n, []),
				"mean_w": (sum(widths) / n, []),
				"mean_l": (sum(lengths) / n, []),
				"num_samples": (n, []),
				}

	def __len__(self) -> int:
		return len(self.images)

	def __getitem__(self, idx: int) -> TensorDict:
		img = self.images[idx]
		if self.img_transform is not None:
			img = self.img_transform(img)
		sample = to_tensordict(img, self.labels[idx], self.splits_str)
		if self.soft_labels is not None:
			sample["soft_labels"] = self.soft_labels[idx]
		return sample
