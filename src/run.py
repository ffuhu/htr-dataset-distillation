import os
import platform

# CTC loss is not available on MPS in current PyTorch versions; allow CPU fallback.
if platform.system() == "Darwin":
	os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from typing import Optional

from omegaconf import DictConfig
from torch import float as torch_float

from hydra import main
from hydra.utils import instantiate
from omegaconf import OmegaConf
from data import DataModule
from code_version import stamp
# from crnn import CRNN_Wrapper

from lightning.pytorch import seed_everything

from contextlib import ExitStack

def load_config(filename: str):
	return OmegaConf.load(filename)

@main(version_base=None, config_path="../configs", config_name="default.yaml")
def _main(cfg: DictConfig):
	code_commit = stamp()  # fails if the code does not match CODE_VERSION
	if "seed" in cfg:
		seed_everything(cfg.seed)

	data_module: DataModule = instantiate(cfg.data_module)

	cfg.max_h = data_module.get_max_h()
	cfg.max_w = data_module.get_max_w()
	cfg.max_l = data_module.get_max_l()
	cfg.t2i, cfg.i2t = data_module.get_dictionaries()
	cfg.vocabulary_size = len(cfg.i2t)

	ckpt_path: Optional[str] = ""
	if "weights" in cfg:
		ckpt_path = cfg.weights
		cfg.model_wrapper._args_.insert(0, ckpt_path)
		cfg.model_wrapper._target_ += ".load_from_checkpoint"
		print(f"Loading weights from {ckpt_path}.")

	model_wrapper = instantiate(cfg.model_wrapper)

	cfg.trainer.min_steps = data_module.get_min_steps()

	trainer = instantiate(cfg.trainer)

	trainer.logger.experiment.config["code_commit"] = code_commit
	trainer.logger.experiment.config["min_steps"] = trainer.min_steps
	trainer.logger.experiment.config["max_steps"] = trainer.max_steps
	trainer.logger.experiment.config["min_epochs"] = trainer.min_epochs
	trainer.logger.experiment.config["max_epochs"] = trainer.max_epochs

	trainer.logger.experiment.config["model"] = OmegaConf.to_container(model_wrapper.model_config, resolve=True, throw_on_missing=True)
	trainer.logger.experiment.config["loss_function"] = OmegaConf.to_container(cfg.loss_function, resolve=True, throw_on_missing=True)
	trainer.logger.experiment.config["optimizer"] = OmegaConf.to_container(cfg.optimizer, resolve=True)
	trainer.logger.experiment.config["t2i"] = OmegaConf.to_container(cfg.t2i, resolve=True)
	trainer.logger.experiment.config["i2t"] = OmegaConf.to_container(cfg.i2t, resolve=True)

	print(model_wrapper.model)
	test_batch = next(iter(data_module.test_dataloader()[0]))
	print(test_batch)
	model_wrapper.print_model_summary(input_batch=test_batch)

	try:
		data_module.set_trainer_data(trainer)
	except:
		pass

	extra = instantiate(cfg.extra)
	with ExitStack() as context_stack:
		extra(cfg, data_module, model_wrapper, trainer, context_stack)

		if cfg.train:
			if "from_checkpoint" in cfg:
				ckpt_path = cfg.from_checkpoint.path
				if "strict_loading" in cfg.from_checkpoint:
					model_wrapper.strict_loading = cfg.from_checkpoint.strict_loading
				print(f"Continuing training from checkpoint file '{ckpt_path}'")
			else:
				ckpt_path = None

			trainer.fit(model_wrapper, data_module, ckpt_path=ckpt_path)

			if "checkpoint_after_training" in cfg:
				path = cfg.checkpoint_after_training.path
				# weights_only still saves:
				# epoch, global_step, pytorch-lightning_version, state_dict, loops
				weights_only = cfg.checkpoint_after_training.weights_only if "weights_only" in cfg.checkpoint_after_training else None
				trainer.save_checkpoint(path, weights_only=weights_only)

			for c in trainer.callbacks:
				if type(c).__name__ == "ModelCheckpoint":
					ckpt_path = c.best_model_path

			if ckpt_path is not None and ckpt_path == "":
				print("Could not find best checkpoint path.")
				if "checkpoint_after_training" in cfg:
					ckpt_path = cfg.checkpoint_after_training.path
				else:
					ckpt_path = None

		print(f"Loading checkpoint from '{ckpt_path}'.")
		trainer.test(model_wrapper, data_module, ckpt_path=ckpt_path, weights_only=False)

if __name__ == "__main__":
	def experiment_id(d, b, c, u):
		components = [d, f"{b}", c, u]
		return '-'.join([c for c in components if c is not None])

	def stats_file(run_type: str, split: str, b, c, u):
		components = [run_type, f"{b}", c, u]
		return '-'.join([c for c in components if c is not None]) + f"_{split}"

	def run_name(s, t, b, c, u):
		components = [f"{b}", c, u]
		name: str = "{s} {run_type} {t}"

		run_type: str = "zero-shot"
		if b is not None:
			run_type = "fine-tuning (" + ' '.join([c for c in components if c is not None]) + ')'

		return name.format(s=s, run_type=run_type, t=t)

	OmegaConf.register_new_resolver("int_div", lambda x, y: x//y)
	OmegaConf.register_new_resolver("mul", lambda x, y: x*y)
	OmegaConf.register_new_resolver("eval", lambda x: eval(x))
	OmegaConf.register_new_resolver("load", load_config)
	OmegaConf.register_new_resolver("experiment_id", experiment_id)
	OmegaConf.register_new_resolver("stats_file", stats_file)
	OmegaConf.register_new_resolver("run_name", run_name)
	_main()
	OmegaConf.clear_resolver("int_div")
	OmegaConf.clear_resolver("mul")
	OmegaConf.clear_resolver("eval")
	OmegaConf.clear_resolver("load")
	OmegaConf.clear_resolver("experiment_id")
	OmegaConf.clear_resolver("stats_file")
	OmegaConf.clear_resolver("run_name")
