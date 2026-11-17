from typing import Optional
from torch import Tensor

from torch.nn import Module, LSTM, Dropout, Linear

class RNN(Module):
	def __init__(self, modules: dict[str, Module], *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.rnn: LSTM
		self.dropout: Dropout
		self.out_projection: Linear

		for m_name, m in modules.items():
			self.register_module(m_name, m)

	def forward(
			self,
			input_seq: Tensor,
			initial_hidden_state: Optional[Tensor] = None,
			initial_cell_state: Optional[Tensor] = None
			) -> tuple[Tensor, Tensor, Tensor]:
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
		- output_seq: Tensor(batch, seq_len, output_size)
		- last_hidden_state: Tensor(D*num_layers, batch, hidden_size)
		- last_cell_state: Tensor(D*num_layers, batch, hidden_size)
		"""

		initial_states = (initial_hidden_state, initial_cell_state) if (initial_hidden_state is not None) and (initial_cell_state is not None) else None

		output_seq, last_states = self.rnn(input_seq, initial_states)
		# output_seq: (batch, seq_len, hidden_size)
		# last_states[0]: (D*num_layers, batch, hidden_size)
		# last_states[1]: (D*num_layers, batch, hidden_size)

		output_seq = self.out_projection(self.dropout(output_seq))
		# output_seq: (batch, seq_len, output_size)

		# (batch, seq_len, output_size), (D*num_layers, batch, hidden_size), (D*num_layers, batch, hidden_size)
		return output_seq, last_states[0], last_states[1]
