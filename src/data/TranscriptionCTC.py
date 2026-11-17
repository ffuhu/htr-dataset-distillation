from typing import Any, Callable, Optional
from torch import Tensor, long as torch_long
from .huggingface import HuggingFaceDataset

from tensordict import TensorDict, stack as stack_tensordicts
from torch import tensor, maximum, zeros, stack as torch_stack
from torch.nn import Sequential

def to_tensordict(img: Tensor, label: Tensor, split: str) -> TensorDict:
	"""
	Assembles the per-sample TensorDict shared by every CTC dataset (real or synthetic):
	width/length are always derived from the tensors' own shapes, never carried as separate
	metadata, so they can't drift out of sync with the data (`img`, `label`) they describe.
	"""
	height: Tensor = tensor(img.shape[1], dtype=torch_long)
	width: Tensor = tensor(img.shape[2], dtype=torch_long)
	length: Tensor = tensor(label.shape[0], dtype=torch_long)

	batch: TensorDict = TensorDict({})

	batch["img"] = img
	batch["label"] = label

	batch["shape"] = tensor([height, width], dtype=torch_long)
	batch["length"] = length
	batch["split"] = split

	return batch

def collate_fn(b: list[TensorDict]) -> TensorDict:
	keys = b[0].keys()
	batch_size = len(b)
	# Pad images and targets
	image_max_shape = tensor(b[0]["img"].shape)
	label_max_shape = tensor(b[0]["label"].shape)
	for s_idx in range(1, len(b)):
		image_max_shape = maximum(image_max_shape, tensor(b[s_idx]["img"].shape))
		label_max_shape = maximum(label_max_shape, tensor(b[s_idx]["label"].shape))

	images = zeros([batch_size] + image_max_shape.tolist(), dtype=b[0]["img"].dtype)
	labels = zeros([batch_size] + label_max_shape.tolist(), dtype=b[0]["label"].dtype)

	for sample_idx, sample in enumerate(b):
		image_slice: list[slice] = [slice(dim_size) for dim_size in sample["img"].shape]
		image_slice.insert(0, slice(sample_idx, sample_idx+1))

		label_slice: list[slice] = [slice(dim_size) for dim_size in sample["label"].shape]
		label_slice.insert(0, slice(sample_idx, sample_idx+1))

		try:
			images[tuple(image_slice)] = sample["img"]
			labels[tuple(label_slice)] = sample["label"]
		except:
			print("Images shape:", images.shape)
			print("Image slice:", image_slice)
			print("Labels shape:", labels.shape)
			print("Label slice:", label_slice)

			raise

	batch = TensorDict({"img": images, "label": labels}, batch_size=[batch_size])
	for k in {key for key in keys if key not in ["img", "label"]}:
		v = [s[k] for s in b]
		if isinstance(v[0], Tensor):
			batch[k] = torch_stack(v)
		elif isinstance(v[0], TensorDict):
			batch[k] = stack_tensordicts(v)
		else:
			batch[k] = v

	return batch

class HFCTC(HuggingFaceDataset):
	def __init__(
			self,
			collate_fn: Callable[[TensorDict], TensorDict],
			root: str,
			data_files: Optional[dict[str, str]] = None,
			splits: list[str] = ["train"],
			img_transform: Optional[Sequential] = None,
			target_transform: Optional[Sequential] = None,
			initial_symbols: dict[str, int] = {},
			special_symbols: list[str] = [],
			map_fn_kwargs: dict = {}, # reduce_ratio: 1.0
			map_kwargs: dict = {},
			prepare_data: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
			stats_file: Optional[str] = None,
			vocabulary_file: Optional[str] = None,
			*args,
			**kwargs
			):
		super().__init__(
				collate_fn,
				root,
				data_files,
				splits,
				img_transform,
				target_transform,
				initial_symbols,
				special_symbols,
				map_fn_kwargs,
				map_kwargs,
				prepare_data,
				stats_file,
				vocabulary_file,
				*args,
				**kwargs
			   )

	def add_control_tokens(self, sequence: list[int]) -> list[int]:
		return sequence

	def __getitem__(self, idx) -> TensorDict:
		sample = self.loadSample(idx)

		img = sample["image"]
		label = sample["transcription"]

		if self.img_transform:
			# (height, width, channels) -> (channels, height, width)
			x: Tensor = self.img_transform(img)
		else:
			x = img

		label = self.add_control_tokens(label)

		if self.target_transform:
			y = self.target_transform(label)
		else:
			y = label

		y = tensor(y, dtype=torch_long)

		return to_tensordict(x, y, self.splits_str)
