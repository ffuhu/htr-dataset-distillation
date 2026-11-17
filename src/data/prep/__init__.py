from .staves import prepare_data as staff_prep
from .htr_lines import prepare_data as htr_lines_prep

def no_prep(sample, *args, **kwargs):
	return sample
