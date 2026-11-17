from typing import Optional
from torch import Tensor
from torch.nn import Module, Embedding
import torch

from torch import\
		tensor,\
		no_grad,\
		tril

from .CNN import CNN
from .RNN import RNN

class CRNN_CTC(Module):
	def __init__(
				self,
				pad: int,
				blank: int,
				max_h: int,
				max_w: int,
				max_l: int,
				vocabulary_size: int,
				channels: int,
				modules: dict[str, Module],
				*args,
				**kwargs
				):
		super().__init__(*args, **kwargs)

		self.pad: Tensor
		self.blank: Tensor

		self.channels: int = channels
		self.vocabulary_size: int = vocabulary_size

		self.max_h: int = max_h
		self.max_w: int = max_w
		self.max_l: int = max_l

		self.encoder: CNN
		self.decoder: RNN

		self.register_buffer("pad", tensor(pad, dtype=torch.long))
		self.register_buffer("blank", tensor(blank, dtype=torch.long))

		for m_name, m in modules.items():
			self.register_module(m_name, m)

	def forward_encoder(self, X: Tensor) -> Tensor:
		"""
		Receives:
		- img: Tensor(batch, channels, height, width)

		Returns:
		- features: Tensor(batch, channels', height', width')
		"""
		return self.encoder(X) # visual_features: (batch, channels', height', width')

	def forward_decoder(
			self,
			input_seq: Tensor,
			initial_hidden_state: Optional[Tensor] = None,
			initial_cell_state: Optional[Tensor] = None
			):
		"""
		With:
			D = 2 if bidirectional else 1
			output_size = output_projection.out_features

		Receives:
		- input_seq: Tensor(batch, seq_len, input_size)
		[
			- initial_hidden_state: Tensor(D*num_layers, batch, hidden_size)
			- initial_cell_state: Tensor(D*num_layers, batch, hidden_size)
		]

		Returns:
		- logits: Tensor(batch, seq_len, output_size)
		- last_hidden_state: Tensor(D*num_layers, batch, hidden_size)
		- last_cell_state: Tensor(D*num_layers, batch, hidden_size)
		"""
		return self.decoder(input_seq, initial_hidden_state, initial_cell_state)

	def forward(
			self,
			X: Tensor,
			initial_hidden_state: Optional[Tensor] = None,
			initial_cell_state: Optional[Tensor] = None
			):
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

		inner = self.forward_encoder(X) # inner: (batch, channels', height', width')
		inner = inner.transpose(1, -1) # Transpose (batch, width', height', channels')
		inner = inner.flatten(-2) # Transpose (batch, width', height' * channels')

		return self.forward_decoder(inner, initial_hidden_state, initial_cell_state)

	@no_grad()
	def predict(
			self,
			X: Tensor,
			initial_hidden_state: Optional[Tensor] = None,
			initial_cell_state: Optional[Tensor] = None
			):
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
		return self.forward(X, initial_hidden_state, initial_cell_state)
