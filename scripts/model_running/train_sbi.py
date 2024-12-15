import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.getcwd())
import ccmodels.modelanalysis.sbi_utils as msbi

#Easy way to select which kind of fit we want
network = "tuning06"

if network == "tuning06":
    features = [f"rate_tuning_{i}" for i in range(8)]
    posterior_path = "data/model/sbi_networks/tuning_cosine0602"
    sims_path = "data/model/cosine_0602"
elif network == "tuning04":
    features = [f"rate_tuning_{i}" for i in range(8)]
    posterior_path = "data/model/sbi_networks/tuning_cosine0402"
    sims_path = "data/model/cosine_0402"



elif network == "tuning06_nochaos":
    features = [f"rate_tuning_{i}" for i in range(8)] + ['indiv_traj_std']
    posterior_path = "data/model/sbi_networks/tuning_nc_cosine0602"
    sims_path = "data/model/cosine_0602"
elif network == "tuning04_nochaos":
    features = [f"rate_tuning_{i}" for i in range(8)]+ ['indiv_traj_std']
    posterior_path = "data/model/sbi_networks/tuning_nc_cosine0402"
    sims_path = "data/model/cosine_0402"

elif network == "pars06":
    features = ['logmean_re', 'mean_cve_dir', 'indiv_traj_std']
    posterior_path = "data/model/sbi_networks/pars_cosine0602"
    sims_path = "data/model/cosine_0602"
elif network == "pars04":
    features = ['logmean_re', 'mean_cve_dir', 'indiv_traj_std']
    posterior_path = "data/model/sbi_networks/pars_cosine0402"
    sims_path = "data/model/cosine_0402"

#Read the data form model simulations and compute real data
parcols = ['J', 'g', 'thetaE', 'sigmaE', 'hEI', 'hII']
params, summary_stats = msbi.get_simulations_summarystats(f"{sims_path}", parcols, features, nsims=None)
summary_data = msbi.get_data_summarystats(features)

#Do the SBI
j0, jf = 0., 4.
g0, gf = 0., 5.
theta0, thetaf = 10., 25.
sigma0, sigmaf = 5., 15.
hei0, heif = 0., 600.
hii0, hiif = 0., 600.
prior, intervals = msbi.setup_prior(j0, jf, g0, gf, theta0, thetaf, sigma0, sigmaf, hei0, heif, hii0, hiif)
posterior = msbi.train_sbi(prior, params, summary_stats)

#Save the results
msbi.save_posterior(f"{posterior_path}", posterior)
                                                         
