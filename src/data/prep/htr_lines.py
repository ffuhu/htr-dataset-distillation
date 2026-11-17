from numpy import array
from cv2 import resize
from PIL import Image as pil_image

def prepare_data(
		sample,
		fixed_height: int | None = 64,
		fixed_width: int | None = None,
		reduce_ratio: float = 1.0,
		tokenize: bool = True,
		*args, **kwargs
		):
	img = sample["image"].convert("RGB")

	old_width = img.width
	old_height = img.height

	img = array(img)

	if (fixed_height is not None):
		height = fixed_height
		if (fixed_width is not None):
			width = fixed_width
		else:
			width = int((height / old_height) * old_width)
	elif (fixed_width is not None):
		width = fixed_width
		height = int((width / old_width) * old_height)
	else:
		width = int(old_width*reduce_ratio)
		height = int(old_height*reduce_ratio)

	img = resize(img, (width, height))

	# print("Image processed")
	sample["image"] = pil_image.fromarray(img)

	# print("Ground truth processed")
	if tokenize:
		sample["transcription"] = list(sample["transcription"])

	return sample
