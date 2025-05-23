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
import ccmodels.dataanalysis.utils as dutl
import ccmodels.dataanalysis.statistics_extraction as ste


N = 4000
kee = 200 
nreps = 10 
filename = "definitive_random_tuned"

def compute_conn_prob(v1_neurons, v1_connections, half=True, n_samps=10):

    #Get the data to be plotted 
    conprob = {}
    conprob["L23"], conprob["L4"] = ste.prob_conn_diffori(v1_neurons, v1_connections, half=half, n_samps=n_samps)
    meandata = {}
    for layer in ["L23", "L4"]:
        p = conprob[layer]
        #Normalize by p(delta=0), which is at index 3
        p.loc[:, ["mean", "std"]] = p.loc[:, ["mean", "std"]] /p.loc[3, "mean"]
        #meandata[layer]  = plotutils.add_symmetric_angle(p['mean'].values)
        meandata[layer]  = p['mean'].values

    return meandata

orionly= True
local_connectivity = False 
mode = 'cosine'
intmode = 'tunedinh'

units, connections, rates = loader.load_data()
connections = fl.remove_autapses(connections)
connections.loc[:, 'syn_volume'] /=  connections.loc[:, 'syn_volume'].mean()

neurons_L23 = fl.filter_neurons(units, layer='L23', tuning='matched')
neurons_L4 = fl.filter_neurons(units, layer='L4', tuning='matched')

rates23 = rates[neurons_L23['id'], :]
rates4  = rates[neurons_L4['id'], :]
tcurvedata = np.mean(dutl.shift_multi(rates23, neurons_L23['pref_ori']), axis=0)
means_data = compute_conn_prob(units, connections)

cvoexp, cvdexp = utl.compute_circular_variance(rates23, orionly=True)

bins = np.linspace(0.,1.,9)
cvdist, _ = np.histogram(cvdexp, bins=bins, density=True)

bins = np.linspace(0.,2.,9)
rdist = np.zeros(8)
rdist_density = np.zeros(8)
nsampled = 200
for nrep in range(100):
    idx = np.random.choice(rates23.shape[0], nsampled)
    distsample, _ = np.histogram(rates23[idx, :].ravel(), bins=bins, density=False)
    rdist += distsample

    distsample, _ = np.histogram(rates23[idx, :].ravel(), bins=bins, density=True)
    rdist_density += distsample

rdist /= 100
rdist_density /= 100


id = 0
if  intmode=='normal':
    best_pars = np.array([[1.32405126e+00, 5.96999824e-01, 8.16734695e+00, 8.55531502e+00,1.45977509e+02, 3.13572388e+02, 4.79209095e-01, 2.37898633e-01]])
    betas = [best_pars[id, 6], best_pars[id, 7], 0., 0., 0., 0.]
elif intmode=='tunedinh':
    #best_pars = np.array([[1.92722714e+00, 5.60854852e-01, 7.13633251e+00, 7.44682407e+00,1.33497223e+02, 3.84251801e+02, 3.04715216e-01, 2.62478530e-01]])
    best_pars = np.array([[2.16435086e+00, 7.02782583e-01, 7.32673001e+00, 7.20601672e+00, 1.39987691e+02, 4.15600751e+02, 5.96647088e-01, 3.83488676e-01]])
    betas = [best_pars[id, 6], best_pars[id, 7], best_pars[id, 6], best_pars[id, 6], best_pars[id, 6], best_pars[id, 7]]
elif intmode=='kin':
    betas = np.zeros(6) 

for i in range(nreps):
    print(f"SIMULATION ID: {i}")

    aE_t, aI_t, re, ri, rx, stdre, units_sample, connections_sample, QJ, n_neurons, original_tuned_ids, original_prefori = md.make_simulation(units, connections, rates, kee, N, best_pars[id,0], best_pars[id,1], hEI=best_pars[id,4], hII=best_pars[id,5],  
                                                                                                                                    theta_E=19., sigma_tE=best_pars[id,2], theta_I=19.0, sigma_tI=best_pars[id,3], cos_b=betas,
                                                                                                                                    mode=mode, local_connectivity=local_connectivity, orionly=orionly, prepath='data')


    utl.write_synthetic_data(f"{filename}_{i}", units_sample, connections_sample, re, ri, rx, original_prefori, prepath="data")



    for reshuffle_mode in ['all', 'L23', 'L4']:
        aE_t, aI_t, re, ri, stdre, units_reshuffle, connections_reshuffle, QJ_reshuffle = md.make_simulation_fixed_structure(units_sample, connections_sample, QJ, rx, n_neurons, theta_E=19., sigma_tE=best_pars[id, 2], hEI=best_pars[id, 4], hII=best_pars[id, 5], theta_I=19., sigma_tI=best_pars[id,3],
                                                                                                                                            orionly=True, reshuffle=reshuffle_mode)

        utl.write_synthetic_data(f'{filename}_{reshuffle_mode[:3]}_{i}', units_reshuffle, connections_reshuffle, re, ri, rx, original_prefori, prepath='data')