import os

from lightning.pytorch import Callback, LightningModule, Trainer
import torch

class ExpertSnapshotter(Callback):
	"""
	MTT: saves a snapshot of the model's TRAINABLE parameters after every training epoch on
	the FULL dataset -- the "expert trajectory buffer" that MTT needs as its parameter-space
	matching target.

	Deliberately lightweight: only `named_parameters()` (no BatchNorm buffers, no optimizer
	state, no Lightning metadata) -- ~0.75MB per snapshot with this model (187K params). A
	full Lightning `ModelCheckpoint` would be ~10-20x heavier per snapshot and would keep only
	the best checkpoint, not the whole trajectory -- hence a separate callback rather than
	reusing `trainer_callbacks.checkpointer`.

	Each file is a plain dict {name: Tensor} that can be loaded directly into
	`torch.func.functional_call` without going through `CRNN_CTC_Wrapper`/Lightning -- the
	entry point used by `distill_mtt_ctc.py`.

	`max_epoch` (disk mitigation): `distill_mtt_ctc.py` only reads epochs up to
	`max_start_epoch + expert_epochs_ahead` of each trajectory (epochs 0-41 with the current
	values) -- saving until actual convergence (~80-180 depending on the dataset) wastes most
	snapshots (IAM/GW/READ2016 already take 1.3-1.6GB each in trajectories alone, almost all
	never used). `max_epoch=None` (default) keeps the original behaviour unchanged.

	`every_n_steps` (E13): one snapshot every M optimizer steps instead of one per epoch
	(including the initial one, step 0). Files keep the name `epoch_NNNN.pt` so that
	`distill_mtt_ctc.ExpertTrajectories` reads them unchanged, but NNNN is then the snapshot
	index (step NNNN*M), not the epoch; `every_n_epochs`/`max_epoch` are ignored.
	"""

	def __init__(self, dirpath: str, every_n_epochs: int = 1, max_epoch: int | None = None,
				every_n_steps: int | None = None):
		super().__init__()
		self.dirpath = dirpath
		self.every_n_epochs = every_n_epochs
		self.max_epoch = max_epoch
		self.every_n_steps = every_n_steps
		os.makedirs(self.dirpath, exist_ok=True)

	def _save(self, pl_module: LightningModule, index: int) -> None:
		snapshot = {name: p.detach().clone().cpu() for name, p in pl_module.model.named_parameters()}
		torch.save(snapshot, os.path.join(self.dirpath, f"epoch_{index:04d}.pt"))

	def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
		if self.every_n_steps is not None:
			assert trainer.global_step == 0, "every_n_steps snapshots assume training starts at step 0."
			self._save(pl_module, 0)

	def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx) -> None:
		if self.every_n_steps is not None and trainer.global_step % self.every_n_steps == 0:
			self._save(pl_module, trainer.global_step // self.every_n_steps)

	def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
		if self.every_n_steps is not None:
			return
		epoch = trainer.current_epoch
		if epoch % self.every_n_epochs != 0:
			return
		if self.max_epoch is not None and epoch > self.max_epoch:
			return

		snapshot = {name: p.detach().clone().cpu() for name, p in pl_module.model.named_parameters()}
		path = os.path.join(self.dirpath, f"epoch_{epoch:04d}.pt")
		torch.save(snapshot, path)
