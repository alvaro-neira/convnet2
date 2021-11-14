import numpy as np
from ssearch import SSearch

ssearch = SSearch('./configs/aneira_tfr.config', "SKETCH")

ssearch.load_features()

ssearch.filenames[:5]