import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import sys
import os 
sys.path.append(os.getcwd())
import argparse

import ccmodels.modelanalysis.model as md 
import ccmodels.modelanalysis.utils as utl
import ccmodels.modelanalysis.sbi_utils as msbi 

import ccmodels.utils.angleutils as au

import ccmodels.dataanalysis.processedloader as loader
import ccmodels.dataanalysis.filters as fl


N = 8000
k = 400 
nreps = 10

parcols =['J', 'g', 'thetaE', 'sigmaE', 'hEI', 'hII'] 
p, s= msbi.get_simulations_summarystats("data/model/cosine_0402_POST/", parcols, ['mean_re', 'mean_cve_dir','indiv_traj_std'] + [f'rate_tuning_{i}' for i in range(8)])
simulations = pd.DataFrame(data=torch.hstack((p, s)), columns=parcols + ['mean_re', 'mean_cve_dir', 'indiv_traj_std']+ [f'rate_tuning_{i}' for i in range(8)])
beta = 0.4
th = 0.26

# Group by the specified columns (J, g, theta, sigma, hEI, hII)
grouped = simulations.groupby(parcols)
# Calculate means and SEM for mean_re and mean_cve_dir
df_means = grouped.mean().reset_index()
df_sem = grouped.sem().reset_index()

mask=(np.abs(df_means['rate_tuning_0']-4.1)/4.1<th)&(np.abs(df_means['rate_tuning_3']-1.1)/1.1<th)&(np.abs(df_means['mean_re']-1.7)/1.7<th)&(np.abs(df_means['mean_cve_dir']-0.23)/0.23<th)
pars = df_means.loc[mask, parcols].values[0]
#pars = df_means.loc[, parcols].values
print(pars.shape)


#pars = [1.3, 1.0, 24.15, 9.33, 100.47, 85.42] 
mode = 'cosine'
filename = "best_search" #'best_ale'

pars = np.array([  2.4183798 ,   2.54088253,  10.02267932,   3.83874689, 93.21315129, 164.0631539 ])

units, connections, rates = loader.load_data()
connections = fl.remove_autapses(connections)
connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

diff_ori = np.empty(0, dtype=int) 

for i in range(nreps):
    print(f"SIMULATION ID: {i}")
    #old pars
    #aE_t, aI_t, re, ri, rx, stdre, units_sample, connections_sample, QJ, n_neurons, original_tuned_ids, original_prefori = md.make_simulation(units, connections, rates, k, N, pars[0], pars[1], hEI=pars[4], hII=pars[5],  
    #                                                                                                                                    theta_E=pars[2], sigma_tE=pars[3], theta_I=pars[2], sigma_tI=pars[3], cos_b=[beta, 0.2],
    #                                                                                                                                    mode=mode, local_connectivity=False, orionly=True)

    #new pars
    aE_t, aI_t, re, ri, rx, stdre, units_sample, connections_sample, QJ, n_neurons, original_tuned_ids, original_prefori = md.make_simulation(units, connections, rates, k, N, pars[0], pars[1], hEI=pars[4], hII=pars[5],  
                                                                                                                                        theta_E=19., sigma_tE=pars[2], theta_I=19., sigma_tI=pars[3], cos_b=[beta, 0.2],
                                                                                                                                        mode=mode, local_connectivity=False, orionly=True)

    utl.write_synthetic_data(f"{filename}_{i}", units_sample, connections_sample, re, ri, rx, original_prefori, prepath="data")


    for reshuffle_mode in ['alltuned', 'L23tuned', 'L4tuned']:
        aE_t, aI_t, re, ri, stdre, units_reshuffle, connections_reshuffle, QJ_reshuffle = md.make_simulation_fixed_structure(units_sample, connections_sample, QJ, rx, n_neurons, theta_E=pars[2], sigma_tE=pars[3], hEI=pars[4], hII=pars[5], theta_I=pars[2], sigma_tI=pars[3],
                                                                                                                                            orionly=True, reshuffle=reshuffle_mode)

        utl.write_synthetic_data(f'{filename}_{reshuffle_mode[:3]}_{i}', units_reshuffle, connections_reshuffle, re, ri, rx, original_prefori, prepath='data')