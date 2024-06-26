import numpy as np
import pandas as pd

import sys
import os 
sys.path.append(os.getcwd())
import ccmodels.modelanalysis.sbi_utils as msbi 

#Easy way to select which kind of fit we want
network = "5pars"

if network == "5pars":
    features = ['mean_re', 'std_re', 'mean_cve_dir', 'std_cve_dir', 'indiv_traj_std']
    posterior_path = "data/model/sbi_networks/k400_disordered_5pars"
    output_path = "scripts/figures/output/panel_5pars"
    sims_path = "data/model/k400_disorder"
elif network == "allpars":
    features = ['mean_re', 'std_re', 'mean_cve_dir', 'std_cve_dir', 'cv_curl23', 'cv_curl4', 'indiv_traj_std']
    posterior_path = "data/model/sbi_networks/k400_disordered_allpars"
    output_path = "scripts/figures/output/panel_allpars"
    sims_path = "data/model/k400_disorder"

#Read the data form model simulations and compute real data
params, summary_stats = msbi.get_simulations_summarystats(f"{sims_path}", features, nsims=None)
summary_data = msbi.get_data_summarystats(features) 

#Do the SBI
prior, intervals = msbi.setup_prior()
posterior = msbi.train_sbi(prior, params, summary_stats)

#Save the results
msbi.save_posterior(f"{posterior_path}", posterior)