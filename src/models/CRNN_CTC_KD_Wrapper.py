from typing import Optional

from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor, tensor, arange, no_grad
from torch.nn.functional import kl_div

from .CRNN_CTC_Wrapper import CRNN_CTC_Wrapper

class CRNN_CTC_KD_Wrapper(CRNN_CTC_Wrapper):
	"""
	Training with the teacher's soft labels (frame-level KD) in addition to CTC. Not used in
	the paper experiments.

		L = kd_alpha * KL(teacher_t || student_t) + (1 - kd_alpha) * L_CTC

	The KL is computed per frame with temperature kd_temperature (standard tau^2 KD scaling)
	and masked to the valid frames of each sample (width // width_reduction, the same quantity
	CTC uses as input_lengths -- derived from the tensor shape, never from separate metadata).
	Requires teacher and student with the same width_reduction (1:1 frame alignment, no
	temporal resampling).

	Two mutually exclusive sources of soft labels:
	- **Offline**: `soft_labels` in the batch -- the teacher's raw LOGITS (batch, T_max, V),
	  precomputed by a relabelling script (not included). Incompatible with augmentation (the
	  SyntheticCTCDataset guard prevents it: the augmented image would be misaligned with the
	  precomputed logits).
	- **Online**: `teacher_ckpt` in the constructor -- the frozen teacher (kept out of the
	  module registry: neither optimizer nor checkpoint) relabels the ALREADY AUGMENTED image
	  in every batch, one extra forward pass without gradient. Alignment is guaranteed by
	  construction: this is the way to combine KD with augmentation.

	With neither source it behaves exactly like CRNN_CTC_Wrapper.
	Validation/test inherit the SER metrics unchanged (the teacher plays no part).
	"""

	def __init__(
				self,
				model: DictConfig,
				loss_function_config: DictConfig,
				optimizer_config: DictConfig,
				t2i: dict[str, int],
				i2t: dict[int, str],
				kd_alpha: float = 0.75,
				kd_temperature: float = 2.0,
				teacher_ckpt: Optional[str] = None,
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
		self.kd_alpha: float = kd_alpha
		self.kd_temperature: float = kd_temperature

		# A list so that Lightning does NOT register it as a submodule: the teacher must not
		# enter the state_dict (it would double the checkpoint) nor the optimizer.
		self._online_teacher: list[CRNN_CTC_Wrapper] = []
		if teacher_ckpt:
			teacher = CRNN_CTC_Wrapper.load_from_checkpoint(teacher_ckpt, map_location="cpu", weights_only=False)
			teacher.eval()
			teacher.freeze()
			assert int(teacher.model.encoder.width_reduction) == int(self.model.encoder.width_reduction), \
				(f"KD online exige mismo width_reduction: teacher "
				f"{teacher.model.encoder.width_reduction} != student {self.model.encoder.width_reduction}")
			self._online_teacher = [teacher]

	def training_step(self, batch: TensorDict, batch_idx: int) -> Optional[Tensor]:
		total_loss = tensor(0.0, device=self.device)
		for _batch in batch:
			X = _batch["img"]
			Y = _batch["label"]

			W = _batch["shape"][:, 1]
			L = _batch["length"]

			logits, *_ = self.model(X)  # (batch, T, V), raw logits

			W_red = W.div(self.model.encoder.width_reduction, rounding_mode="trunc")

			log_probs = logits.permute(1, 0, 2).log_softmax(dim=-1)
			ctc_loss: Tensor = self.loss_function(log_probs, Y, W_red, L)

			teacher_logits: Optional[Tensor] = None
			if "soft_labels" in _batch.keys():
				teacher_logits = _batch["soft_labels"]  # (batch, T_max, V), precomputed
			elif self._online_teacher:
				online = self._online_teacher[0]
				if online.device != self.device:
					online.to(self.device)
				with no_grad():
					teacher_logits, *_ = online.model(X)  # (batch, T, V), on the already augmented X

			if teacher_logits is not None:
				tau = self.kd_temperature

				T = min(logits.shape[1], teacher_logits.shape[1])
				student = (logits[:, :T] / tau).log_softmax(dim=-1)
				teacher = (teacher_logits[:, :T] / tau).log_softmax(dim=-1)

				# per-sample mask of valid frames (same criterion as input_lengths)
				mask = arange(T, device=self.device).unsqueeze(0) < W_red.unsqueeze(1)

				kl_per_frame = kl_div(student, teacher, log_target=True, reduction="none").sum(dim=-1)
				kd_loss = (kl_per_frame * mask).sum() / mask.sum().clamp(min=1) * tau ** 2

				loss = self.kd_alpha * kd_loss + (1.0 - self.kd_alpha) * ctc_loss

				self.log("train/kd_loss", kd_loss.item(), on_epoch=True, batch_size=X.size(0))
				self.log("train/ctc_loss", ctc_loss.item(), on_epoch=True, batch_size=X.size(0))
			else:
				loss = ctc_loss

			self.log("train/loss", loss.item(), on_epoch=True, prog_bar=True, batch_size=X.size(0))
			total_loss += loss

		return total_loss / len(batch)
