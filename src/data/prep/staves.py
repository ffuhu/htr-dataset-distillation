from typing import Optional

from numpy import array
from cv2 import resize
from PIL import Image as pil_image

def prepare_data(
		sample,
		fixed_height: Optional[int] = None, # 64
		fixed_width: Optional[int] = None,
		):
	img = sample["image"].convert("RGB")

	old_width = img.width
	old_height = img.height

	img = array(sample['image'])

	if (fixed_height is not None):
		height = fixed_height
		if (fixed_width is not None):
			width = fixed_width
		else:
			width = int((height / old_height) * old_width)
	elif (fixed_width is not None):
		width = fixed_width
		height = int((width / old_height) * old_height)
	else:
		width = old_width
		height = old_height

	img = resize(img, (width, height))

	# print("Image processed")

	ground_truth = sample["transcription"].strip("\n").split("\n")

	# print("Ground truth processed")

	sample["transcription"] = ground_truth
	sample["image"] = pil_image.fromarray(img)

	return sample
