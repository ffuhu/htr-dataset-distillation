from torch import Tensor

from torch.nn import Module

class CNN(Module):
	def __init__(
			self,
			width_reduction: int,
			height_reduction: int,
			in_channels: int,
			out_channels: int,
			modules: dict[str, Module],
			*args,
			**kwargs
			):
		super().__init__(*args, **kwargs)

		self.width_reduction: int = width_reduction
		self.height_reduction: int = height_reduction
		self.in_channels: int = in_channels
		self.out_channels: int = out_channels

		for m_name, m in modules.items():
			self.register_module(m_name, m)

	def forward(self, img: Tensor) -> Tensor:
		"""
		Receives:
		- img: Tensor(batch, channels, height, width)

		Returns:
		- features: Tensor(batch, channels', height', width')
		"""

		last_hidden_state = img

		for i, layer in enumerate(self.children()):
			last_hidden_state = layer(last_hidden_state)
			# last_hidden_state: (batch, channels_i, height_i, width_i)

		# (batch, channels', height', width')
		return last_hidden_state
