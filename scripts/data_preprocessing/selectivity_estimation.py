''''This script extracts the oreintation and direction tuning properties of neurons in Layer 2/3 and 4 or V1
    NOTE: run this within the scripts folder'''

import gc
import os
import shutil
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from ccmodels.preprocessing.connectomics import client_version, subset_v1l234
from ccmodels.preprocessing.selectivity import orientation_extractor, von_mises, von_mises_single, is_selective, cell_area_identifiers
