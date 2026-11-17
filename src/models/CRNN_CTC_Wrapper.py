from typing import Optional
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import\
		Tensor,\
		dtype as torch_dtype
from torch.nn import CrossEntropyLoss

from torch import\
		tensor,\
		no_grad,\
		zeros as torch_zeros

from collections import defaultdict
from torchinfo import summary

from utils import SER, greedy_decoding
from .ModelWrapper import ModelWrapper
from .others import CRNN_CTC

class CRNN_CTC_Wrapper(ModelWrapper[CRNN_CTC,CrossEntropyLoss]):
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
		super().__init__(
				model,
				loss_function_config,
				optimizer_config,
				t2i,
				i2t,
				*args,
				**kwargs
				)

	def forward(self, *X: Tensor):
		"""
		Receives:
		- img: Tensor(batch, channels, height, width)
		[
			- initial_hidden_state: Tensor(D*num_layers, batch, hidden_size)
			- initial_cell_state: Tensor(D*num_layers, batch, hidden_size)
		]

		Returns:
		- logits: Tensor(batch, seq_len, output_size)
		- last_hidden_state: Tensor(D*num_layers, batch, hidden_size)
		- last_cell_state: Tensor(D*num_layers, batch, hidden_size)
		"""
		return self.model(*X)

	@no_grad()
	def print_model_summary(self, input_batch: Optional[TensorDict] = None, input_sizes: Optional[list[list[int]]] = None, dtypes: Optional[list[torch_dtype]] = None):
		if input_sizes is not None:
			assert dtypes is not None
			assert len(dtypes) == len(input_sizes)
			inputs = [
					torch_zeros(input_sizes[0], dtype=dtypes[0], device=self.device),
					torch_zeros(input_sizes[1], dtype=dtypes[1], device=self.device),
					tensor([[input_sizes[0][2], input_sizes[0][3]]*input_sizes[0][0]], dtype=dtypes[1], device=self.device)
					]
		else:
			assert input_batch is not None
			assert "img" in input_batch
			inputs = [
					input_batch["img"].to(device=self.device),
					]

		summary(self, input_data=inputs, dtypes=dtypes)

	def on_train_epoch_start(self):
		super().on_train_epoch_start()
		self.model.train()

	def training_step(self, batch: TensorDict, batch_idx: int) -> Optional[Tensor]:
		total_loss = tensor(0.0, device=self.device)
		for _batch in batch:
			# Get inputs and labels
			X = _batch["img"]
			Y = _batch["label"]

			W = _batch["shape"][:, 1]
			L = _batch["length"]

			# print(_batch)
			# print(f"{W=};{L=}")

			logits, *_ = self.model(X) # Forward pass

			# Prepare everything for the loss function

			# logits: (batch, length, vocabulary_size)
			# (batch, length, vocabulary_size) -> (length, batch, vocabulary_size)
			logits = logits.permute(1, 0, 2).log_softmax(dim=-1)
			W = W.div(self.model.encoder.width_reduction, rounding_mode="trunc")

			# CTCLoss receives:
			# - log_probs: Tensor(length, batch, vocabulary)
			# - targets: Tensor(batch, length)
			# - input_lengths: (batch)
			# - target_lengths: (batch)
			# And returns:
			# - output: (1) if reduction is not 'none', (batch) otherwise
			loss: Tensor = self.loss_function(logits, Y, W, L)

			# Log loss value
			self.log("train/loss", loss.item(), on_epoch=True, prog_bar=True, batch_size=X.size(0))

			# Accumulate loss
			total_loss += loss

		# Average loss over dataloaders
		return total_loss / len(batch)

	def on_validation_epoch_start(self):
		super().on_validation_epoch_start()

		self.model.eval()

		self.validation_sample_id: dict[str, int] = defaultdict(int)

		self.total_symbol_distance: dict[str, int] = defaultdict(int)
		self.total_symbol_length: dict[str, int] = defaultdict(int)

	def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Optional[Tensor]:
		super().validation_step(batch, batch_idx)

		X = batch["img"]
		Y = batch["label"]

		W = batch["shape"][:, 1]
		L = batch["length"]

		splits = batch["split"]

		logits, *_ = self.model.predict(X)
		# logits: (batch, length, vocabulary_size)

		prediction = logits.softmax(dim=-1)

		W = W.div(self.model.encoder.width_reduction, rounding_mode="trunc")

		for sample_idx in range(logits.size(0)):
			prediction = greedy_decoding(logits[sample_idx, :W[sample_idx]], ignore_index=self.t2i["<CTC>"]).tolist()
			sample_Y = Y[sample_idx, :L[sample_idx]].tolist()

			s, sl, ser = SER(prediction, sample_Y) # Already prepared for SER calculation

			sample_split: str = splits[sample_idx]

			self.validation_sample_id[sample_split] += 1

			self.total_symbol_distance[sample_split] += s
			self.total_symbol_length[sample_split] += sl

	def on_validation_epoch_end(self):
		super().on_validation_epoch_end()

		for k in self.validation_sample_id.keys():
			self.log(f"{k}/symbol edit distance", self.total_symbol_distance[k], on_epoch=True)
			self.log(f"{k}/symbol length", self.total_symbol_length[k], on_epoch=True)
			self.log(f"{k}/SER", 100*self.total_symbol_distance[k]/self.total_symbol_length[k], on_epoch=True, prog_bar=True)

		self.validation_sample_id: dict[str, int] = defaultdict(int)

		self.total_symbol_distance: dict[str, int] = defaultdict(int)
		self.total_symbol_length: dict[str, int] = defaultdict(int)

	def on_test_epoch_start(self):
		super().on_test_epoch_start()

		self.model.eval()
		# logger.experiment.summary
		self.logger.experiment.define_metric("total symbol edit distance", step_metric="epoch")
		self.logger.experiment.define_metric("total symbol length", step_metric="epoch")
		self.logger.experiment.define_metric("total SER", step_metric="epoch")

		self.test_sample_id: dict[str, int] = defaultdict(int)

		self.total_symbol_distance: dict[str, int] = defaultdict(int)
		self.total_symbol_length: dict[str, int] = defaultdict(int)

	def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Optional[Tensor]:
		super().test_step(batch, batch_idx)

		X = batch["img"]
		Y = batch["label"]

		W = batch["shape"][:, 1]
		L = batch["length"]

		splits = batch["split"]

		logits, *_ = self.model.predict(X)
		# logits: (batch, max_length, vocabulary_size)
		# weights: list<dict<str, list<Tensor>>>
		# predicted_lengths: (batch)
		preds = logits.argmax(dim=-1)
		confidences = logits.softmax(dim=-1).max(dim=-1)[0]

		W = W.div(self.model.encoder.width_reduction, rounding_mode="trunc")

		for sample_idx in range(len(preds)):
			prediction = greedy_decoding(logits[sample_idx, :W[sample_idx]], ignore_index=self.t2i["<CTC>"]).tolist()
			sample_Y = Y[sample_idx, :L[sample_idx]].tolist()

			s, sl, ser = SER(prediction, sample_Y) # Already prepared for SER calculation

			sample_split: str = splits[sample_idx]

			self.test_sample_id[sample_split] += 1

			self.total_symbol_distance[sample_split] += s
			self.total_symbol_length[sample_split] += sl

			metrics = {
				"sample/split": sample_split,
				"sample/idx": self.test_sample_id[sample_split],
				"sample/confidences": confidences.tolist(),
				"sample/prediction": "".join([self.i2t[token] for token in prediction]),
				"sample/target": "".join([self.i2t[token] for token in sample_Y]),
				"sample/SER": ser,
				}

			self.logger.log_metrics(metrics)

	def on_test_epoch_end(self):
		super().on_test_epoch_end()

		for k in self.test_sample_id.keys():
			self.logger.experiment.summary[f"{k}/total symbol edit distance"] = self.total_symbol_distance[k]
			self.logger.experiment.summary[f"{k}/total symbol length"] = self.total_symbol_length[k]
			self.logger.experiment.summary[f"{k}/total SER"] = 100*self.total_symbol_distance[k]/self.total_symbol_length[k]

		self.test_sample_id: dict[str, int] = defaultdict(int)

		self.total_symbol_distance: dict[str, int] = defaultdict(int)
		self.total_symbol_length: dict[str, int] = defaultdict(int)
