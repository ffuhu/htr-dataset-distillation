from torch import no_grad

def default(cfg, data_module, model_wrapper, trainer, context_stack, *args, **kwargs):
	return

def bias_against_blank(cfg, data_module, model_wrapper, trainer, context_stack, blank_bias: float = -5.0, *args, **kwargs):
	"""
	Intervention against CTC blank collapse (blank_bias=-5.0 in all paper runs): initializes
	the output projection's bias for the blank/<CTC> index to a strongly negative value, so
	the model starts with a low prior on blank instead of whatever a generic Linear init
	happens to assign it. This does not change
	the loss or the model architecture -- just the starting point of one bias entry.
	"""
	blank_index = model_wrapper.t2i["<CTC>"]
	with no_grad():
		model_wrapper.model.decoder.out_projection.bias[blank_index] = blank_bias
	print(f"[extras.bias_against_blank] Set out_projection.bias[{blank_index}] (<CTC>) = {blank_bias}")
