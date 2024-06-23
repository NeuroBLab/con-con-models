import numpy as np
import pandas as pd

import sys
sys.path.append("/home/victor/Fisica/Research/Milan/con-con-models/")
import ccmodels.modelanalysis.sbi_utils as msbi 

features = ['mean_re', 'std_re', 'mean_cve_dir', 'std_cve_dir', 'indiv_traj_std']
params, summary_stats = msbi.get_simulations_summarystats("k400_disorder", features, nsims=None)
summary_data = msbi.get_data_summarystats(features, prepath="../../data")
prior, intervals = msbi.setup_prior()

posterior = msbi.train_sbi(prior, params, summary_stats)

msbi.save_posterior("../../data/model/sbi_networks/k400_disordered_5pars", posterior)