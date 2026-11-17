from typing import Any, Callable, Optional
from tensordict import TensorDict

from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule

class MyDataset(Dataset):
	def __init__(
			self,

			collate_fn: Callable[[TensorDict], TensorDict],
			vocabulary: dict[str, int] = dict(),

			batch_size: int = 1,
			shuffle: bool = False,

			num_workers: int = 0,
			persistent_workers: bool = False,

			*args,
			**kwargs
			) -> None:
		super().__init__(*args, **kwargs)

		self.collate_fn: Callable[[TensorDict], TensorDict] = collate_fn
		self.batch_size: int = batch_size

		self.class_to_int: dict[str, int] = {k: v for k, v in vocabulary.items()}
		self.int_to_class: dict[int, str] = {v: k for k, v in vocabulary.items()}

		self.samples: Any

		self.shuffle: bool = shuffle

		self.num_workers: int = num_workers
		self.persistent_workers: bool = persistent_workers

		self.stats: dict[str, tuple[int | float, list[str]]] = dict()

	def add_class(
			self,
			className: str,
			t2i: Optional[dict[str, int]] = None,
			i2t: Optional[dict[int, str]] = None
			):
		if ((t2i is None) and (i2t is not None))\
		or ((t2i is not None) and (i2t is None)):
			print("Cannot use alternative dictionaries if one of them is missing:", end=" ")
			print("t2i: Missing" if t2i is None else "t2i: Given", end=", ")
			print("i2t: Missing" if i2t is None else "i2t: Given")
			t2i = None
			i2t = None

		if t2i is None:
			if className not in self.class_to_int:
				idx = len(self.int_to_class)
				self.class_to_int[className] = idx
				self.int_to_class[idx] = className
		else:
			assert t2i is not None
			assert i2t is not None

			if className not in t2i:
				idx = len(t2i)
				t2i[className] = idx
				i2t[idx] = className

	def get_min_steps(self) -> int:
		return 0

class DataModule(LightningDataModule):
	def __init__(
			self,
			train_datasets: dict[str, MyDataset] = dict(),
			val_datasets: dict[str, MyDataset] = dict(),
			test_datasets: dict[str, MyDataset] = dict(),
			*args,
			**kwargs
			):
		super().__init__(*args, **kwargs)

		self.train_datasets: dict[str, MyDataset] = train_datasets
		self.val_datasets: dict[str, MyDataset] = val_datasets
		self.test_datasets: dict[str, MyDataset] = test_datasets

	def get_max_h(self) -> int:
		return int(max([
			*[ds.stats["max_h"][0] for ds in self.train_datasets.values()],
			*[ds.stats["max_h"][0] for ds in self.val_datasets.values()],
			*[ds.stats["max_h"][0] for ds in self.test_datasets.values()],
			]))

	def get_max_w(self) -> int:
		return int(max([
			*[ds.stats["max_w"][0] for ds in self.train_datasets.values()],
			*[ds.stats["max_w"][0] for ds in self.val_datasets.values()],
			*[ds.stats["max_w"][0] for ds in self.test_datasets.values()],
			]))

	def get_max_l(self) -> int:
		return int(max([
			*[ds.stats["max_l"][0] for ds in self.train_datasets.values()],
			*[ds.stats["max_l"][0] for ds in self.val_datasets.values()],
			*[ds.stats["max_l"][0] for ds in self.test_datasets.values()],
			]))

	def get_min_steps(self) -> int:
		return int(max([
			*[ds.get_min_steps() for ds in self.train_datasets.values()],
			*[ds.get_min_steps() for ds in self.val_datasets.values()],
			*[ds.get_min_steps() for ds in self.test_datasets.values()],
			]))

	def get_dictionaries(self):
		if len(self.val_datasets):
			ds = next(iter(self.val_datasets.values()))
		elif len(self.test_datasets):
			ds = next(iter(self.test_datasets.values()))
		else:
			ds = next(iter(self.train_datasets.values()))

		return ds.class_to_int, ds.int_to_class

	def set_trainer_data(self, trainer):
		for ds_name in self.train_datasets.keys():
			try:
				self.train_datasets[ds_name].set_trainer_data(trainer)
			except:
				pass
		for ds_name in self.val_datasets.keys():
			try:
				self.val_datasets[ds_name].set_trainer_data(trainer)
			except:
				pass
		for ds_name in self.test_datasets.keys():
			try:
				self.test_datasets[ds_name].set_trainer_data(trainer)
			except:
				pass

	def train_dataloader(self):
		return [
				DataLoader(
					ds,
					shuffle=ds.shuffle,
					batch_size=ds.batch_size, collate_fn=ds.collate_fn,
					num_workers=ds.num_workers, persistent_workers=ds.persistent_workers,
					) for ds in self.train_datasets.values()
				]

	def val_dataloader(self):
		return [
				DataLoader(
					ds,
					shuffle=ds.shuffle,
					batch_size=ds.batch_size, collate_fn=ds.collate_fn,
					num_workers=ds.num_workers, persistent_workers=ds.persistent_workers,
					) for ds in self.val_datasets.values()
				]

	def test_dataloader(self):
		return [
				DataLoader(
					ds,
					shuffle=ds.shuffle,
					batch_size=ds.batch_size, collate_fn=ds.collate_fn,
					num_workers=ds.num_workers, persistent_workers=ds.persistent_workers,
					) for ds in self.test_datasets.values()
				]
