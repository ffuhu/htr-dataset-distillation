from typing import Optional, Generic, TypeVar
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import\
		Tensor,\
		dtype as torch_dtype
from torch.nn import Module
from torch.optim import Optimizer

from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger

from torch import\
		tensor,\
		no_grad,\
		zeros as torch_zeros

from hydra.utils import instantiate

from collections import defaultdict
from torchinfo import summary

Model = TypeVar("Model", bound=Module)
Loss = TypeVar("Loss", bound=Module)

class ModelWrapper(LightningModule, Generic[Model,Loss]):
	def __init__(
				self,
				model: DictConfig,
				loss_function_config: DictConfig,
				optimizer_config: DictConfig,
				t2i: dict[str, int],
				i2t: dict[int, str],
				*args,
				**kwargs
				):
		super().__init__(*args, **kwargs)

		self.logger: WandbLogger
		self.model_config: DictConfig = model
		self.model: Model = instantiate(self.model_config)
		self.loss_function: Loss = instantiate(loss_function_config)
		self.optimizer: Optimizer = instantiate(optimizer_config, params=self.model.parameters())

		self.t2i: dict[str, int] = t2i
		self.i2t: dict[int, str] = i2t

		self.save_hyperparameters(logger=False)

	def configure_optimizers(self):
		return [{
				"optimizer": self.optimizer,
				},]

	@no_grad()
	def print_model_summary(self, input_batch: Optional[TensorDict] = None, input_sizes: Optional[list[list[int]]] = None, dtypes: Optional[list[torch_dtype]] = None):
		if input_sizes is not None:
			assert dtypes is not None
			assert len(dtypes) == len(input_sizes)
			inputs = [torch_zeros(in_size, dtype=dtype, device=self.device) for in_size, dtype in zip(input_sizes, dtypes)]
		else:
			inputs = input_batch

		summary(self, input_data=inputs, dtypes=dtypes)

	def on_train_epoch_start(self):
		super().on_train_epoch_start()
		self.model.train()

	def forward(self, *X: Tensor):
		"""
		Receives:
		- X: any amount of Tensor(batch, *)

		Returns:
		- output: Any
		"""
		return self.model(*X)

	def training_step(self, batch: TensorDict, batch_idx: int) -> Optional[Tensor]:
		pass

	def on_validation_epoch_start(self):
		super().on_validation_epoch_start()

		self.model.eval()
		# self.validation_sample_id: dict[str, int] = defaultdict(int)
		# self.total_distance: dict[str, int] = defaultdict(int)
		# self.total_length: dict[str, int] = defaultdict(int)

	def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Optional[Tensor]:
		super().validation_step(batch, batch_idx)

	def on_validation_epoch_end(self):
		super().on_validation_epoch_end()

		# Log the metrics
		# for k in self.total_distance.keys():
			# self.log(f"{k}/edit distance", self.total_distance[k], on_epoch=True)
			# self.log(f"{k}/length", self.total_length[k], on_epoch=True)
			# self.log(f"{k}/SER", 100.0*self.total_distance[k]/self.total_length[k], on_epoch=True, prog_bar=True)

		# self.validation_sample_id: dict[str, int] = defaultdict(int)
		# self.total_distance: dict[str, int] = defaultdict(int)
		# self.total_length: dict[str, int] = defaultdict(int)

	def on_test_epoch_start(self):
		super().on_test_epoch_start()

		self.model.eval()
		# logger.experiment.summary
		# self.logger.experiment.define_metric("total edit distance", step_metric="epoch")
		# self.logger.experiment.define_metric("total length", step_metric="epoch")
		# self.logger.experiment.define_metric("total SER", step_metric="epoch")

		# self.test_sample_id: dict[str, int] = defaultdict(int)
		# self.total_distance: dict[str, int] = defaultdict(int)
		# self.total_length: dict[str, int] = defaultdict(int)

	def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Optional[Tensor]:
		super().test_step(batch, batch_idx)

	def on_test_epoch_end(self):
		super().on_test_epoch_end()

		# Log the metrics
		# for k in self.total_distance.keys():
			# self.logger.experiment.summary[f"{k}/total edit distance"] = self.total_distance[k]
			# self.logger.experiment.summary[f"{k}/total length"] = self.total_length[k]
			# self.logger.experiment.summary[f"{k}/total SER"] = 100.0*self.total_distance[k]/self.total_length[k]

		# self.test_sample_id: dict[str, int] = defaultdict(int)
		# self.total_distance: dict[str, int] = defaultdict(int)
		# self.total_length: dict[str, int] = defaultdict(int)
