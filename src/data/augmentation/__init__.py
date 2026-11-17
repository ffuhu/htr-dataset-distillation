from torch import float as torch_float

from torch.nn import Sequential
from torchvision.transforms.v2 import ToDtype, RandomApply, RandomAffine, PILToTensor, Grayscale

def no_aug(grayscale: bool = False, *args, **kwargs):
	augmentations = [
			PILToTensor(),
			ToDtype(torch_float)
			]

	if grayscale:
		augmentations.insert(-2, Grayscale())

	return Sequential(*augmentations)

def no_aug_tensor(*args, **kwargs):
	"""
	Identity for buffers that are already float tensors in HFCTC's post-transform space
	(SyntheticCTCDataset): no PILToTensor/Grayscale/ToDtype needed.
	"""
	return Sequential()

def full_pipeline_tensor(
		translation: tuple[float, float] = (.05, .05),
		rotation_degrees: tuple[float, float] = (-3.0, 3.0),
		transform_chance: float = .2,
		*args, **kwargs
		):
	"""
	Tensor-native counterpart of full_pipeline for SyntheticCTCDataset buffers (already
	float (C,H,W) tensors): the same RandomAffine with the same parameters, applied directly
	to the tensor -- torchvision v2 transforms accept tensors natively. Restores the
	augmentation symmetry with the real-line baselines (pipeline-controls finding:
	RandomAffine is worth ~15-16 CER points at wr8/K=100). Only for HARD-label buffers:
	augmenting images that carry per-frame soft
	labels would desynchronize them (SyntheticCTCDataset enforces this).
	"""
	return Sequential(
			RandomApply([RandomAffine(degrees=rotation_degrees, translate=translation)], transform_chance),
			)

def full_pipeline(
		translation: tuple[float, float] = (-.05, .05),
		rotation_degrees: tuple[float, float] = (-3.0, 3.0),
		transform_chance: float = .2,
		grayscale: bool = False,
		*args, **kwargs
		):
	augmentations = [
			RandomApply([RandomAffine(degrees=rotation_degrees, translate=translation)], transform_chance),
			PILToTensor(),
			ToDtype(torch_float)
			]

	if grayscale:
		augmentations.insert(-2, Grayscale())

	return Sequential(*augmentations)
