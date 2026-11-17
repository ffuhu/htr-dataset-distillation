from typing import Any, Callable, Optional
from torch import Tensor
from datasets import DatasetDict

from tensordict import TensorDict
from torch.nn import Sequential
from .base import MyDataset
from functools import partial
from copy import deepcopy

from datasets import load_dataset, concatenate_datasets, Dataset

from rich.progress import track

import pathlib
import yaml

def allow_indices(sample, index: int, allowed_indices: list[int]) -> bool:
	return index in allowed_indices

def forbid_indices(sample, index: int, forbidden_indices: list[int]) -> bool:
	return not index in forbidden_indices

class HuggingFaceDataset(MyDataset):
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

				filter_func: Optional[Callable] = None,

				*args,
				**kwargs
				):
		super().__init__(collate_fn, vocabulary=initial_symbols, *args, **kwargs)

		self.vocabulary_needed: bool = False
		self.stats_needed: bool = False

		self.splits: list[str] = splits

		self.img_transform: Optional[Sequential] = img_transform
		self.target_transform: Optional[Sequential] = target_transform

		# Load dataset and build vocabulary
		if data_files is not None:
			data_files = dict(**data_files)

		ds = load_dataset(path=root, data_files=data_files)

		assert isinstance(ds, DatasetDict)
		self.initialize_stats()

		if prepare_data:
			# ds.cleanup_cache_files()
			# print(map_fn_kwargs)
			ds = ds.map(prepare_data, fn_kwargs=dict(**map_fn_kwargs), **map_kwargs)

		# Create vocabulary
		ds_name = root.split("/")[-1]
		self.splits_str: str = "_".join(self.splits)

		if vocabulary_file is None:
			vocabulary_file = f"{ds_name}/{self.splits_str}"
		else:
			vocabulary_file = vocabulary_file.replace(".yaml", "").replace("-c2i", "").replace("-i2c", "")

		self.vocabulary_file: str = vocabulary_file

		if stats_file is None:
			stats_file = f"{ds_name}/{self.splits_str}"
		else:
			stats_file = stats_file.replace(".yaml", "").replace("-stats", "")

		self.stats_file: str = stats_file

		try:
			self.load_vocabulary()
		except FileNotFoundError:
			self.vocabulary_needed = True

		try:
			self.load_stats()
		except FileNotFoundError:
			self.stats_needed = True

		if self.vocabulary_needed:
			self.get_vocabulary(ds, update=True, save=True)

		# Keep only requested splits
		self.samples: Dataset = concatenate_datasets([ds[s] for s in self.splits])

		if filter_func is not None:
			self.samples = self.samples.filter(filter_func, with_indices=True)

		if self.stats_needed:
			self.get_stats(ds, update=True, save=True)

		self.special_indices: list[int] = [self.class_to_int[s] for s in special_symbols if s in self.class_to_int]
		self.non_special_indices: list[int] = [i for i in self.int_to_class.keys() if i not in self.special_indices]

	def extract_symbols(self, sample) -> list[str]:
		return sample["transcription"]

	def get_vocabulary(self, ds: DatasetDict, update: bool = False, save: bool = False) -> tuple[dict[str, int], dict[int, str]]:
		if save and not update:
			print("Cannot save vocabulary without updating. Saving step will be skipped.")

		t2i: dict[str, int] = {k: v for k, v in self.class_to_int.items()}
		i2t: dict[int, str] = {k: v for k, v in self.int_to_class.items()}

		for _split, samples in ds.items():
			for sample_idx, sample in enumerate(track(samples, description=f"Adding symbols from {_split} split samples to vocabulary")):
				l = self.extract_symbols(sample)

				for symbol in l:
					self.add_class(symbol, t2i=t2i, i2t=i2t)

		if update:
			self.class_to_int = deepcopy(t2i)
			self.int_to_class = deepcopy(i2t)

			if save:
				self.save_vocabulary()

		return t2i, i2t

	def save_vocabulary(self, update_safe_indices: bool = False):
		pathlib.Path("data/"+"/".join(self.vocabulary_file.split("/")[:-1])).mkdir(parents=True, exist_ok=True)

		with open(f"data/{self.vocabulary_file}-c2i.yaml", "w") as file:
			yaml.safe_dump(self.class_to_int, file)
		with open(f"data/{self.vocabulary_file}-i2c.yaml", "w") as file:
			yaml.safe_dump(self.int_to_class, file)

		if update_safe_indices:
			self.non_special_indices = [i for i in self.int_to_class.keys() if i not in self.special_indices]

	def load_vocabulary(self, update_safe_indices: bool = False):
		with open(f"data/{self.vocabulary_file}-c2i.yaml", "r") as file:
			c2i = yaml.safe_load(file)
		with open(f"data/{self.vocabulary_file}-i2c.yaml", "r") as file:
			i2c = yaml.safe_load(file)

		self.class_to_int = c2i
		self.int_to_class = i2c

		if update_safe_indices:
			self.non_special_indices = [i for i in self.int_to_class.keys() if i not in self.special_indices]

	def initialize_stats(self, stats: Optional[dict[str, tuple[int | float, list[str]]]] = None):
		if stats is not None:
			stats["max_h"] = (0, list())
			stats["max_w"] = (0, list())
			stats["max_l"] = (0, list())
			stats["min_h"] = (99999, list())
			stats["min_w"] = (99999, list())
			stats["min_l"] = (99999, list())
			stats["mean_h"] = (.0, list())
			stats["mean_w"] = (.0, list())
			stats["mean_l"] = (.0, list())
			stats["num_samples"] = (.0, list())
			return

		self.stats["max_h"] = (0, list())
		self.stats["max_w"] = (0, list())
		self.stats["max_l"] = (0, list())
		self.stats["min_h"] = (99999, list())
		self.stats["min_w"] = (99999, list())
		self.stats["min_l"] = (99999, list())
		self.stats["mean_h"] = (.0, list())
		self.stats["mean_w"] = (.0, list())
		self.stats["mean_l"] = (.0, list())
		self.stats["num_samples"] = (.0, list())

	def get_stats(self, ds, update: bool = False, save: bool = False) -> dict[str, tuple[int | float, list[str]]]:
		if save and not update:
			print("Cannot save stats without updating. Saving step will be skipped.")

		num_samples = 0
		mean_h = .0
		mean_w = .0
		mean_l = .0

		stats: dict[str, tuple[int | float, list[str]]] = dict()
		self.initialize_stats(stats)

		for split in self.splits:
			for sample_idx, sample in enumerate(track(ds[split], description=f"Calculating stats of dataset samples ({split})")):
				l = sample["transcription"]

				*_, w, h = sample["image"].size

				if h > stats["max_h"][0]:
					stats["max_h"] = (h, [f"{split}-{sample_idx}"])
				elif h == stats["max_h"][0]:
					stats["max_h"][1].append(f"{split}-{sample_idx}")

				if h < stats["min_h"][0]:
					stats["min_h"] = (h, [f"{split}-{sample_idx}"])
				elif h == stats["min_h"][0]:
					stats["min_h"][1].append(f"{split}-{sample_idx}")

				if w > stats["max_w"][0]:
					stats["max_w"] = (w, [f"{split}-{sample_idx}"])
				elif w == stats["max_w"][0]:
					stats["max_w"][1].append(f"{split}-{sample_idx}")

				if w < stats["min_w"][0]:
					stats["min_w"] = (w, [f"{split}-{sample_idx}"])
				elif w == stats["min_w"][0]:
					stats["min_w"][1].append(f"{split}-{sample_idx}")

				if len(l) > stats["max_l"][0]:
					stats["max_l"] = (len(l), [f"{split}-{sample_idx}"])
				elif len(l) == stats["max_l"][0]:
					stats["max_l"][1].append(f"{split}-{sample_idx}")

				if len(l) < stats["min_l"][0]:
					stats["min_l"] = (len(l), [f"{split}-{sample_idx}"])
				elif len(l) == stats["min_l"][0]:
					stats["min_l"][1].append(f"{split}-{sample_idx}")

				mean_h += h
				mean_w += w
				mean_l += len(l)
				num_samples += 1

		stats["mean_h"] = (float(mean_h)/num_samples, list())
		stats["mean_w"] = (float(mean_w)/num_samples, list())
		stats["mean_l"] = (float(mean_l)/num_samples, list())
		stats["num_samples"] = (num_samples, list())

		if update:
			print("Updating stats...")
			self.stats = deepcopy(stats)
			print("Stats updated...")

			if save:
				print("Saving stats...")
				self.save_stats()
				print(f"Stats saved to data/{self.stats_file}-stats.yaml")

		return stats

	def save_stats(self):
		pathlib.Path("data/"+"/".join(self.stats_file.split("/")[:-1])).mkdir(parents=True, exist_ok=True)

		with open(f"data/{self.stats_file}-stats.yaml", "w") as file:
			yaml.safe_dump(self.stats, file)

	def load_stats(self):
		with open(f"data/{self.stats_file}-stats.yaml", "r") as file:
			stats = yaml.safe_load(file)

		self.stats = stats

	def __len__(self) -> int:
		return len(self.samples)

	def loadSample(self, idx: int):
		sample = self.samples[idx]

		sample["transcription"] = [self.class_to_int[t] for t in sample["transcription"]]

		return sample
